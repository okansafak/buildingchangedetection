import os
import time
import json
import csv
import uuid
import base64
from io import BytesIO, StringIO
from flask import Flask, render_template, request, jsonify, send_file, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from model.change_detector import BuildingChangeDetector
from model.live_satellite import LiveSatelliteFetcher, WAYBACK_RELEASES, LIVE_HOTSPOTS, capture_info, placeholder_tile_mask
from model.project_manager import ProjectManager
from PIL import Image
from model.ml_detector import MLChangeDetector, load_rgb
from model.segmenter import ModelUnavailable
from model.jobs import JobRunner
from model import geo, georef

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
CORS(app)

detector = BuildingChangeDetector()
ml_detector = MLChangeDetector()  # the ONNX model is downloaded/loaded on first use
jobs = JobRunner()
live_fetcher = LiveSatelliteFetcher()
project_mgr = ProjectManager()

UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = os.path.join(app.root_path, 'static', 'results')
MAX_UPLOAD_MB = 512  # two full-resolution aerial mosaics (e.g. 2 x 8192 x 4468 GeoTIFF)
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024

MODEL_ERROR = "Yapay zeka modeli yüklenemedi, internet bağlantısını kontrol edin: {}"

# Wayback patches are always analysed at zoom 17 (~0.9 m/px, 2x2 tiles ≈ 450 m). The map picker
# sends its view zoom; a zoomed-out view (e.g. 14, ~7 m/px) would leave no buildings to detect.
LIVE_ANALYSIS_ZOOM = 17


# Cache last detection results for download
LAST_RESULTS = {}


@app.errorhandler(413)
def upload_too_large(_error):
    return jsonify({"success": False, "error": f"Dosyalar çok büyük (en fazla {MAX_UPLOAD_MB} MB)."}), 413


def _uploaded_path(path):
    """Real path of an existing file inside UPLOAD_FOLDER, else None (clients send server paths)."""
    if not path:
        return None
    folder = os.path.realpath(app.config['UPLOAD_FOLDER'])
    real = os.path.realpath(path)
    try:
        inside = os.path.commonpath([folder, real]) == folder
    except ValueError:  # different drives on Windows
        return None
    return real if inside and os.path.isfile(real) else None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return Response(status=204)

@app.route('/api/live/hotspots', methods=['GET'])
def get_live_hotspots():
    """Returns list of curated real-world urban change hotspots."""
    return jsonify(list(LIVE_HOTSPOTS.values()))

@app.route('/api/live/years', methods=['GET'])
def get_live_years():
    """Returns available historical Wayback years."""
    years = []
    for y, info in WAYBACK_RELEASES.items():
        years.append({"year": y, "title": info["title"], "date": info["date"]})
    return jsonify(years)


def _jpeg_data_uri(image):
    buf = BytesIO()
    image.convert('RGB').save(buf, 'JPEG', quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode('ascii')


MAX_LIVE_GRID = 8  # tiles per side: 8 x 8 zoom-17 tiles ≈ 1.8 km


def _grid_size(data):
    """Tiles per side of the live patch (default 2); ValueError with a Turkish message when invalid."""
    value = data.get('grid_size')
    if value is None:
        return 2
    if isinstance(value, bool) or not isinstance(value, (int, str)) or not str(value).isdigit():
        raise ValueError(f"Analiz alanı 1–{MAX_LIVE_GRID} karo arasında olmalı.")
    grid = int(value)
    if not 1 <= grid <= MAX_LIVE_GRID:
        raise ValueError(f"Analiz alanı 1–{MAX_LIVE_GRID} karo arasında olmalı.")
    return grid


@app.route('/api/live/preview', methods=['POST'])
def live_preview():
    """T1/T2 Wayback imagery for step 2 only (tiles are cached); detection runs later via /api/live/detect."""
    data = request.json or {}
    lat = float(data.get('lat', 41.1070))
    lon = float(data.get('lon', 28.7900))
    year_t1 = str(data.get('year_t1', '2014'))
    year_t2 = str(data.get('year_t2', '2026'))
    try:
        grid = _grid_size(data)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    try:
        img_t1, img_t2, bounds, gsd = live_fetcher.fetch_bitemporal_pair(
            lat=lat, lon=lon, zoom=LIVE_ANALYSIS_ZOOM, year_t1=year_t1, year_t2=year_t2, grid_size=grid
        )
    except Exception as e:
        return jsonify({"success": False, "error": f"Canlı uydu verisi çekilirken hata oluştu: {str(e)}"}), 500
    missing_t1 = _missing_tile_count(img_t1)
    missing_t2 = _missing_tile_count(img_t2)
    return jsonify({
        "success": True,
        "t1": _jpeg_data_uri(img_t1),
        "t2": _jpeg_data_uri(img_t2),
        "bounds": bounds,
        "gsd": gsd,
        "grid_size": grid,
        "years": {"t1": year_t1, "t2": year_t2},
        "capture": {"t1": capture_info(lat, lon, year_t1), "t2": capture_info(lat, lon, year_t2)},
        "missing_tiles": {"t1": missing_t1, "t2": missing_t2, "total": grid * grid},
    })


def _missing_tile_count(image):
    return int(placeholder_tile_mask(image)[::256, ::256].sum())


class LiveImageryMissing(ValueError):
    """The archive has no imagery at all for one of the dates (Turkish message)."""


def _live_detect(params, report=None):
    """Fetch the Wayback pair for params and run the chosen engine; returns the result dict."""
    report = report or (lambda fraction, stage: None)
    start_time = time.time()
    report(0.01, f"Uydu karoları indiriliyor ({params['grid_size']}×{params['grid_size']} karo)")
    img_t1, img_t2, bounds, gsd = live_fetcher.fetch_bitemporal_pair(
        lat=params['lat'], lon=params['lon'], zoom=LIVE_ANALYSIS_ZOOM,
        year_t1=params['year_t1'], year_t2=params['year_t2'], grid_size=params['grid_size'],
    )
    mask_t1, mask_t2 = placeholder_tile_mask(img_t1), placeholder_tile_mask(img_t2)
    if mask_t1.all() or mask_t2.all():
        raise LiveImageryMissing("Seçilen yıllar için bu bölgede arşiv görüntüsü yok; farklı bir yıl veya konum deneyin.")
    missing = mask_t1 | mask_t2
    warnings = []
    if missing.any():
        n_missing = int(missing[::256, ::256].sum())
        warnings.append(f"{n_missing}/{params['grid_size'] ** 2} karo seçilen yıllardan birinde arşivde yok; "
                        "bu alanlar analiz dışı bırakıldı.")
    if params['engine'] == 'ml':
        tiles_ref = georef.from_bounds_wgs84(bounds, img_t2.width, img_t2.height, source="tiles")
        result = ml_detector.detect(img_t1, img_t2, georef=tiles_ref, min_area_m2=params['min_area_m2'],
                                    mode="fast", progress=report, invalid_mask=missing if missing.any() else None)
    else:
        result = detector.detect(
            img1=img_t1,
            img2=img_t2,
            ground_truth=None,
            threshold=params['threshold'],
            min_area_m2=params['min_area_m2'],
            gsd=gsd,
            bounds=bounds
        )
    result["inference_time_sec"] = round(time.time() - start_time, 2)
    result["center"] = [params['lat'], params['lon']]
    result["mode"] = "live_satellite"
    result["grid_size"] = params['grid_size']
    result["years"] = {"t1": params['year_t1'], "t2": params['year_t2']}
    result["capture"] = {"t1": capture_info(params['lat'], params['lon'], params['year_t1']),
                         "t2": capture_info(params['lat'], params['lon'], params['year_t2'])}
    result["warnings"] = warnings
    LAST_RESULTS['latest'] = result
    return result


@app.route('/api/live/detect', methods=['POST'])
def run_live_detect():
    """
    Fetches real-time multi-temporal satellite imagery from Esri Wayback
    for any global coordinates and executes the building change detection engine.
    grid_size (1-8 tiles per side, default 2) sets the analysed area. With engine "ml"
    and job: true it runs as a background job (202 + job_id, poll /api/jobs/<id>);
    otherwise synchronously.
    """
    data = request.json or {}
    try:
        grid = _grid_size(data)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    params = {
        'lat': float(data.get('lat', 41.1070)),
        'lon': float(data.get('lon', 28.7900)),
        # the request's 'zoom' is the map view zoom, not an analysis scale
        'year_t1': str(data.get('year_t1', '2014')),
        'year_t2': str(data.get('year_t2', '2026')),
        'threshold': float(data.get('threshold', 0.45)),
        'min_area_m2': float(data.get('min_area_m2', 40.0)),
        'engine': data.get('engine', 'classic'),
        'grid_size': grid,
    }

    if params['engine'] == 'ml' and data.get('job'):
        def work(report):
            try:
                return _live_detect(params, report)
            except ModelUnavailable as e:
                raise RuntimeError(MODEL_ERROR.format(e)) from e
            except LiveImageryMissing as e:
                raise RuntimeError(str(e)) from e
        return jsonify({"success": True, "job_id": jobs.submit(work)}), 202

    try:
        return jsonify(_live_detect(params))
    except LiveImageryMissing as e:
        return jsonify({"success": False, "error": str(e)}), 422
    except ModelUnavailable as e:
        return jsonify({"success": False, "error": MODEL_ERROR.format(e)}), 503
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Canlı uydu verisi çekilirken hata oluştu: {str(e)}"}), 500

@app.route('/api/detect', methods=['POST'])
def run_detect():
    data = request.json or {}
    if data.get('engine', 'classic') == 'ml':
        return _run_detect_ml(data)
    scenario_id = data.get('scenario_id', 'custom')
    threshold = float(data.get('threshold', 0.45))
    min_area_m2 = float(data.get('min_area_m2', 30.0))
    gsd = float(data.get('gsd', 0.5))

    custom_center = data.get('center') # [lat, lon]

    if scenario_id == 'custom':
        path_A = _uploaded_path(data.get('path_A'))
        path_B = _uploaded_path(data.get('path_B'))
        path_label = None
        center = custom_center if custom_center else [41.0082, 28.9784] # Istanbul default
        if not path_A or not path_B:
            return jsonify({"success": False, "error": "Geçerli T1 ve T2 yüklenen görüntüleri bulunamadı."}), 400
    else:
        return jsonify({"success": False, "error": f"Bilinmeyen senaryo: {scenario_id}"}), 404

    # Classic engine: bounds of a 256x256 patch around center (legacy behaviour, pinned by snapshot tests)
    lat, lon = center[0], center[1]
    bounds = geo.bounds_from_center(lat, lon, 256, 256, gsd)

    start_time = time.time()
    result = detector.detect(
        img1=path_A,
        img2=path_B,
        ground_truth=path_label,
        threshold=threshold,
        min_area_m2=min_area_m2,
        gsd=gsd,
        bounds=bounds
    )
    elapsed = round(time.time() - start_time, 2)
    result["inference_time_sec"] = elapsed
    result["center"] = center
    result["scenario_id"] = scenario_id

    # Store for export
    LAST_RESULTS['latest'] = result

    return jsonify(result)


def _resolve_georef(path_A, path_B, manual):
    """(GeoRef of T2's grid or None, GSD override or None) from sidecars/GeoTIFF tags and manual entry.
    Raises georef.GeoRefError with a Turkish message."""
    epsg = manual.get('epsg') or None
    gsd = manual.get('gsd')
    if gsd not in (None, ''):
        gsd = float(gsd)
        if not 0.01 <= gsd <= 100:
            raise georef.GeoRefError("GSD 0.01–100 m/piksel aralığında olmalı.")
    else:
        gsd = None
    size_A = georef.image_size(path_A)
    size_B = georef.image_size(path_B)
    if manual.get('bounds'):
        # Manual bounds describe the pair after T1 is resized onto T2's grid
        return georef.from_bounds_wgs84(manual['bounds'], *size_B), gsd
    ref_A = georef.read_for_image(path_A, epsg)
    ref_B = georef.read_for_image(path_B, epsg)
    return georef.pair_reference(ref_A, size_A, ref_B, size_B), gsd


def _run_detect_ml(data):
    """Validates the request, then runs MLChangeDetector as a background job (202 + job_id)."""
    scenario_id = data.get('scenario_id', 'custom')
    min_area_m2 = float(data.get('min_area_m2', 30.0))
    mode = data.get('analysis_mode', 'fast')
    if mode not in ('fast', 'deep'):
        return jsonify({"success": False, "error": f"Bilinmeyen analiz modu: {mode}"}), 400

    if scenario_id == 'custom':
        path_A = _uploaded_path(data.get('path_A'))
        path_B = _uploaded_path(data.get('path_B'))
        if not path_A or not path_B:
            return jsonify({"success": False, "error": "Geçerli T1 ve T2 yüklenen görüntüleri bulunamadı."}), 400
        try:
            pair_ref, gsd = _resolve_georef(path_A, path_B, data.get('manual_georef') or {})
        except (georef.GeoRefError, ValueError) as e:
            return jsonify({"success": False, "error": str(e)}), 400
    else:
        return jsonify({"success": False, "error": f"Bilinmeyen senaryo: {scenario_id}"}), 404

    run_id = uuid.uuid4().hex[:12]
    out_dir = os.path.join(app.config['RESULTS_FOLDER'], run_id)

    def work(report):
        start_time = time.time()
        try:
            result = ml_detector.detect(
                path_A, path_B, georef=pair_ref, gsd=gsd, min_area_m2=min_area_m2, mode=mode,
                progress=report,
                out_dir=out_dir, url_prefix=f"/static/results/{run_id}",
            )
        except ModelUnavailable as e:
            raise RuntimeError(MODEL_ERROR.format(e)) from e
        result["inference_time_sec"] = round(time.time() - start_time, 2)
        result["scenario_id"] = scenario_id
        LAST_RESULTS['latest'] = result
        return result

    return jsonify({"success": True, "job_id": jobs.submit(work)}), 202


@app.route('/api/jobs/<job_id>', methods=['GET'])
def get_job(job_id):
    job = jobs.get(job_id)
    if job is None:
        return jsonify({"success": False, "error": "İş bulunamadı."}), 404
    return jsonify({
        "success": True,
        "status": job["status"],      # running | done | error
        "progress": job["progress"],  # 0..1
        "stage": job["stage"],
        "result": job["result"] if job["status"] == "done" else None,
        "error": job["error"],
    })


def _preview_url(path, fname):
    """URL the browser can display: TIFFs get a JPEG preview (longest side 2048 px) next to them."""
    if not path.lower().endswith(georef.GEOTIFF_EXTS):
        return f"/static/uploads/{fname}"
    preview = Image.fromarray(load_rgb(path))
    preview.thumbnail((2048, 2048))
    preview.save(path + ".preview.jpg", quality=88)
    return f"/static/uploads/{fname}.preview.jpg"


def _save_sidecars(image_path, suffix):
    """Stores optional world file / .prj uploads (fields world_<suffix>, prj_<suffix>) next to the image."""
    stem = os.path.splitext(image_path)[0]
    world = request.files.get(f'world_{suffix}')
    if world and world.filename:
        ext = os.path.splitext(world.filename)[1].lower()
        if ext not in georef.WORLD_FILE_EXTS:
            raise georef.GeoRefError(f"Geçersiz world file uzantısı: {ext} (desteklenen: {', '.join(georef.WORLD_FILE_EXTS)})")
        world.save(stem + ext)
    prj = request.files.get(f'prj_{suffix}')
    if prj and prj.filename:
        prj.save(stem + '.prj')


@app.route('/api/upload', methods=['POST'])
def upload_images():
    if 'image_t1' not in request.files or 'image_t2' not in request.files:
        return jsonify({"success": False, "error": "Her iki görüntü (T1 ve T2) gereklidir."}), 400

    f1 = request.files['image_t1']
    f2 = request.files['image_t2']

    if f1.filename == '' or f2.filename == '':
        return jsonify({"success": False, "error": "Seçilen dosya adı geçersiz."}), 400

    ts = int(time.time())
    fname1 = f"t1_{ts}_{secure_filename(f1.filename)}"
    fname2 = f"t2_{ts}_{secure_filename(f2.filename)}"

    save_path1 = os.path.join(app.config['UPLOAD_FOLDER'], fname1)
    save_path2 = os.path.join(app.config['UPLOAD_FOLDER'], fname2)

    f1.save(save_path1)
    f2.save(save_path2)
    try:
        _save_sidecars(save_path1, 't1')
        _save_sidecars(save_path2, 't2')
    except georef.GeoRefError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    georef_A, georef_A_error = georef.describe(save_path1)
    georef_B, georef_B_error = georef.describe(save_path2)

    return jsonify({
        "success": True,
        "path_A": save_path1,
        "path_B": save_path2,
        "url_A": _preview_url(save_path1, fname1),
        "url_B": _preview_url(save_path2, fname2),
        "georef_A": georef_A,
        "georef_B": georef_B,
        "georef_A_error": georef_A_error,
        "georef_B_error": georef_B_error,
    })

@app.route('/api/export/geojson', methods=['GET'])
def export_geojson():
    if 'latest' not in LAST_RESULTS or not LAST_RESULTS['latest'].get('geojson'):
        return jsonify({"error": "Henüz tespit sonucu yok."}), 404
        
    geojson_str = json.dumps(LAST_RESULTS['latest']['geojson'], indent=2, ensure_ascii=False)
    return Response(
        geojson_str,
        mimetype="application/geo+json",
        headers={"Content-Disposition": "attachment;filename=bina_degisim_poligonlari.geojson"}
    )

@app.route('/api/export/csv', methods=['GET'])
def export_csv():
    if 'latest' not in LAST_RESULTS or not LAST_RESULTS['latest'].get('geojson'):
        return jsonify({"error": "Henüz tespit sonucu yok."}), 404
        
    features = LAST_RESULTS['latest']['geojson'].get('features', [])
    si = StringIO()
    writer = csv.writer(si)
    writer.writerow(["Bina_ID", "Degisim_Turu", "Degisim_Tanimi", "Alan_m2", "Cevre_m", "Guven_Skoru_Yuzde", "Enlem", "Boylam", "Piksel_X", "Piksel_Y"])

    def cell(value):
        return '' if value is None else value

    for f in features:
        props = f.get('properties', {})
        centroid = props.get('centroid') or ['', '']  # None when the image is not georeferenced
        centroid_px = props.get('centroid_px') or ['', '']
        writer.writerow([
            cell(props.get('id')),
            cell(props.get('type')),
            cell(props.get('type_tr')),
            cell(props.get('area_m2')),
            cell(props.get('perimeter_m')),
            cell(props.get('confidence_pct')),
            centroid[0],
            centroid[1],
            centroid_px[0],
            centroid_px[1],
        ])
        
    return Response(
        si.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=bina_degisim_raporu.csv"}
    )

# Project Management API Routes
@app.route('/api/projects', methods=['GET'])
def get_projects():
    search = request.args.get('search', '')
    projects = project_mgr.list_projects(search=search)
    return jsonify({"success": True, "projects": projects})

@app.route('/api/projects', methods=['POST'])
def save_project():
    data = request.json or {}
    name = data.get('name', '').strip()
    if not name:
        name = f"Bina Değişim Analizi - {time.strftime('%d.%m.%Y %H:%M')}"
    data['name'] = name
    
    saved = project_mgr.save_project(data)
    # Also update LAST_RESULTS if it has results_data
    if data.get('results_data'):
        LAST_RESULTS['latest'] = data['results_data']
    return jsonify({"success": True, "project": saved})

@app.route('/api/projects/<project_id>', methods=['GET'])
def get_project_by_id(project_id):
    proj = project_mgr.get_project(project_id)
    if not proj:
        return jsonify({"success": False, "error": "Proje bulunamadı"}), 404
    # Set LAST_RESULTS so CSV and GeoJSON downloads work for this project
    if proj.get('results_data'):
        LAST_RESULTS['latest'] = proj['results_data']
    return jsonify({"success": True, "project": proj})

@app.route('/api/projects/<project_id>', methods=['DELETE'])
def delete_project_by_id(project_id):
    deleted = project_mgr.delete_project(project_id)
    if not deleted:
        return jsonify({"success": False, "error": "Proje bulunamadı veya silinemedi"}), 404
    return jsonify({"success": True, "message": "Proje başarıyla silindi"})

@app.route('/api/projects/clear', methods=['POST'])
def clear_all_projects():
    project_mgr.clear_all()
    return jsonify({"success": True, "message": "Tüm projeler başarıyla sıfırlandı"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Building Change Detection Server on http://127.0.0.1:{port} ...")
    app.run(host='0.0.0.0', port=port, debug=False)
