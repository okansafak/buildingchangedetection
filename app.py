import os
import time
import json
import csv
import math
import numpy as np
from io import StringIO
from flask import Flask, render_template, request, jsonify, send_file, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from model.change_detector import BuildingChangeDetector
from model.live_satellite import LiveSatelliteFetcher, WAYBACK_RELEASES, LIVE_HOTSPOTS
from model.project_manager import ProjectManager

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
CORS(app)

detector = BuildingChangeDetector()
live_fetcher = LiveSatelliteFetcher()
project_mgr = ProjectManager()

UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB max

# Preloaded scenarios based on satellite change detection datasets (LEVIR-CD, DSIFN)
SCENARIOS = {
    "levir1": {
        "id": "levir1",
        "title": "LEVIR-CD #1: Yeni Kentsel Konut Gelişimi (Teksas, ABD)",
        "dataset": "LEVIR-CD Benchmark",
        "resolution": "0.5m / piksel",
        "description": "Boş arazide yeni inşa edilen müstakil yerleşim birimleri ve kentsel yayılma.",
        "path_A": "static/samples/levir1/A.png",
        "path_B": "static/samples/levir1/B.png",
        "path_label": "static/samples/levir1/label.png",
        "center": [30.2750, -97.7400], # Austin, TX
        "gsd": 0.5,
        "zoom": 17
    },
    "levir2": {
        "id": "levir2",
        "title": "LEVIR-CD #2: Banliyö Büyümesi ve Yeni Konutlar",
        "dataset": "LEVIR-CD Benchmark",
        "resolution": "0.5m / piksel",
        "description": "Yeni parselasyon ve hızla inşa edilen konut blokları.",
        "path_A": "static/samples/levir2/A.png",
        "path_B": "static/samples/levir2/B.png",
        "path_label": "static/samples/levir2/label.png",
        "center": [32.7767, -96.7970], # Dallas, TX
        "gsd": 0.5,
        "zoom": 17
    },
    "levir3": {
        "id": "levir3",
        "title": "LEVIR-CD #3: Yoğun Yapılaşma ve Çatı Değişimleri",
        "dataset": "LEVIR-CD Benchmark",
        "resolution": "0.5m / piksel",
        "description": "Gelişmekte olan yerleşim alanında yeni çatı ve bina eklentileri.",
        "path_A": "static/samples/levir3/A.png",
        "path_B": "static/samples/levir3/B.png",
        "path_label": "static/samples/levir3/label.png",
        "center": [29.7604, -95.3698], # Houston, TX
        "gsd": 0.5,
        "zoom": 17
    },
    "dsifn1": {
        "id": "dsifn1",
        "title": "DSIFN #1: Sanayi ve Ticari Tesis İnşaatı (Asya Metropolü)",
        "dataset": "DSIFN Benchmark",
        "resolution": "0.6m / piksel",
        "description": "Geniş ölçekli fabrika ve depo binalarının kurulumu.",
        "path_A": "static/samples/dsifn1/A.png",
        "path_B": "static/samples/dsifn1/B.png",
        "path_label": "static/samples/dsifn1/label.png",
        "center": [30.5928, 114.3055], # Wuhan
        "gsd": 0.6,
        "zoom": 17
    },
    "dsifn2": {
        "id": "dsifn2",
        "title": "DSIFN #2: Kentsel Altyapı ve Konut Blokları",
        "dataset": "DSIFN Benchmark",
        "resolution": "0.6m / piksel",
        "description": "Yüksek katlı kentsel genişleme ve yeni blok inşaatı.",
        "path_A": "static/samples/dsifn2/A.png",
        "path_B": "static/samples/dsifn2/B.png",
        "path_label": "static/samples/dsifn2/label.png",
        "center": [39.9042, 116.4074], # Beijing
        "gsd": 0.6,
        "zoom": 17
    }
}

# Cache last detection results for download
LAST_RESULTS = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return Response(status=204)

@app.route('/api/scenarios', methods=['GET'])
def get_scenarios():
    items = []
    for sid, sc in SCENARIOS.items():
        items.append({
            "id": sc["id"],
            "title": sc["title"],
            "dataset": sc["dataset"],
            "resolution": sc["resolution"],
            "description": sc["description"],
            "center": sc["center"],
            "gsd": sc["gsd"],
            "zoom": sc["zoom"],
            "thumb_A": f"/{sc['path_A']}",
            "thumb_B": f"/{sc['path_B']}"
        })
    return jsonify(items)

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

@app.route('/api/live/detect', methods=['POST'])
def run_live_detect():
    """
    Fetches real-time multi-temporal satellite imagery from Esri Wayback
    for any global coordinates and executes the building change detection engine.
    """
    data = request.json or {}
    lat = float(data.get('lat', 41.1070))
    lon = float(data.get('lon', 28.7900))
    zoom = int(data.get('zoom', 17))
    year_t1 = str(data.get('year_t1', '2014'))
    year_t2 = str(data.get('year_t2', '2026'))
    threshold = float(data.get('threshold', 0.45))
    min_area_m2 = float(data.get('min_area_m2', 40.0))
    
    start_time = time.time()
    try:
        # Fetch live bi-temporal satellite imagery
        img_t1, img_t2, bounds, gsd = live_fetcher.fetch_bitemporal_pair(
            lat=lat,
            lon=lon,
            zoom=zoom,
            year_t1=year_t1,
            year_t2=year_t2,
            grid_size=2 # 512x512 px mosaic (~500m x 500m)
        )
        
        # Run building change detection
        result = detector.detect(
            img1=img_t1,
            img2=img_t2,
            ground_truth=None,
            threshold=threshold,
            min_area_m2=min_area_m2,
            gsd=gsd,
            bounds=bounds
        )
        
        elapsed = round(time.time() - start_time, 2)
        result["inference_time_sec"] = elapsed
        result["center"] = [lat, lon]
        result["mode"] = "live_satellite"
        result["years"] = {"t1": year_t1, "t2": year_t2}
        
        LAST_RESULTS['latest'] = result
        return jsonify(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Canlı uydu verisi çekilirken hata oluştu: {str(e)}"}), 500

@app.route('/api/detect', methods=['POST'])
def run_detect():
    data = request.json or {}
    scenario_id = data.get('scenario_id', 'levir1')
    threshold = float(data.get('threshold', 0.45))
    min_area_m2 = float(data.get('min_area_m2', 30.0))
    gsd = float(data.get('gsd', 0.5))
    use_gt = bool(data.get('use_gt', True))
    
    custom_center = data.get('center') # [lat, lon]
    
    if scenario_id in SCENARIOS:
        sc = SCENARIOS[scenario_id]
        path_A = os.path.join(app.root_path, sc['path_A'])
        path_B = os.path.join(app.root_path, sc['path_B'])
        path_label = os.path.join(app.root_path, sc['path_label']) if use_gt else None
        center = custom_center if custom_center else sc['center']
        gsd = sc.get('gsd', gsd)
    elif scenario_id == 'custom':
        path_A = data.get('path_A')
        path_B = data.get('path_B')
        path_label = None
        center = custom_center if custom_center else [41.0082, 28.9784] # Istanbul default
        if not path_A or not os.path.exists(path_A) or not path_B or not os.path.exists(path_B):
            return jsonify({"success": False, "error": "Geçerli T1 ve T2 yüklenen görüntüleri bulunamadı."}), 400
    else:
        return jsonify({"success": False, "error": f"Bilinmeyen senaryo: {scenario_id}"}), 404

    # Calculate geographic bounds around center
    lat, lon = center[0], center[1]
    span_meters = 256.0 * gsd
    lat_delta = span_meters / 111320.0
    lon_delta = span_meters / (111320.0 * np.cos(np.radians(lat)))
    
    bounds = [
        lat - lat_delta / 2,
        lon - lon_delta / 2,
        lat + lat_delta / 2,
        lon + lon_delta / 2
    ]
    
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
    
    return jsonify({
        "success": True,
        "path_A": save_path1,
        "path_B": save_path2,
        "url_A": f"/static/uploads/{fname1}",
        "url_B": f"/static/uploads/{fname2}"
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
    writer.writerow(["Bina_ID", "Degisim_Turu", "Degisim_Tanimi", "Alan_m2", "Cevre_m", "Guven_Skoru_Yuzde", "Enlem", "Boylam"])
    
    for f in features:
        props = f.get('properties', {})
        writer.writerow([
            props.get('feature_id', ''),
            props.get('change_type', ''),
            props.get('type_tr', ''),
            props.get('area_m2', ''),
            props.get('perimeter_m', ''),
            props.get('confidence_pct', ''),
            props.get('centroid', [0, 0])[0],
            props.get('centroid', [0, 0])[1]
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
    print("Starting Building Change Detection Server on http://127.0.0.1:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=False)
