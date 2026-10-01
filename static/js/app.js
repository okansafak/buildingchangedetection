/**
 * Atlas GeoChange - Building Change Detection & Project Workspace
 * Clean Atlas UI Theme + Pure Image Swipe Comparison Engine
 * L.CRS.Simple Pixel Space Engine
 */

// ==========================================
// TOAST NOTIFICATION SYSTEM
// ==========================================
const TOAST_ICONS = {
    success: 'fa-solid fa-circle-check',
    error: 'fa-solid fa-circle-exclamation',
    warning: 'fa-solid fa-triangle-exclamation',
    info: 'fa-solid fa-circle-info'
};

function showToast(message, type = 'info', duration = 3500) {
    const container = document.getElementById('atlas-toast-container');
    if (!container) { console.log(`[Toast ${type}] ${message}`); return; }

    const toast = document.createElement('div');
    toast.className = `atlas-toast toast-${type}`;
    toast.innerHTML = `
        <i class="atlas-toast-icon ${TOAST_ICONS[type] || TOAST_ICONS.info}"></i>
        <span class="atlas-toast-text">${message}</span>
        <button class="atlas-toast-close" onclick="this.parentElement.classList.add('toast-leaving'); setTimeout(() => this.parentElement.remove(), 300);">
            <i class="fa-solid fa-xmark"></i>
        </button>
    `;
    container.appendChild(toast);

    setTimeout(() => {
        if (toast.parentElement) {
            toast.classList.add('toast-leaving');
            setTimeout(() => toast.remove(), 300);
        }
    }, duration);
}

function showConfirm({ title = 'Emin misiniz?', message = '', icon = 'danger', confirmText = 'Evet', cancelText = 'İptal' }) {
    return new Promise((resolve) => {
        const backdrop = document.createElement('div');
        backdrop.className = 'atlas-confirm-backdrop';

        const iconClass = icon === 'danger' ? 'fa-solid fa-trash-can'
            : icon === 'warning' ? 'fa-solid fa-triangle-exclamation'
            : 'fa-solid fa-circle-info';

        backdrop.innerHTML = `
            <div class="atlas-confirm-card">
                <div class="atlas-confirm-icon icon-${icon}">
                    <i class="${iconClass}"></i>
                </div>
                <div class="atlas-confirm-title">${title}</div>
                <div class="atlas-confirm-message">${message}</div>
                <div class="atlas-confirm-actions">
                    <button class="btn btn-secondary btn-sm" id="atlas-confirm-cancel">${cancelText}</button>
                    <button class="btn btn-${icon === 'danger' ? 'danger' : 'primary'} btn-sm" id="atlas-confirm-ok">${confirmText}</button>
                </div>
            </div>
        `;
        document.body.appendChild(backdrop);

        const cleanup = (result) => {
            backdrop.style.opacity = '0';
            backdrop.style.transition = 'opacity 0.2s ease';
            setTimeout(() => backdrop.remove(), 200);
            resolve(result);
        };

        backdrop.querySelector('#atlas-confirm-cancel').addEventListener('click', () => cleanup(false));
        backdrop.querySelector('#atlas-confirm-ok').addEventListener('click', () => cleanup(true));
        backdrop.addEventListener('click', (e) => { if (e.target === backdrop) cleanup(false); });
    });
}

// Application State
const state = {
    // Screen Navigation ('dashboard' or 'workspace')
    activeScreen: 'dashboard',
    currentProjectId: null,
    currentProjectName: 'Yeni Bina Değişim Projesi',
    projectsList: [],
    
    // Wizard Steps (1: Select, 2: Preview, 3: Results)
    currentStep: 1,
    sourceType: 'live-hotspot',
    
    // Live Hotspots & Map Selection (Step 1)
    hotspots: {},
    currentHotspotId: 'istanbul_fikirtepe',
    selectedCoords: [40.9902, 29.0520],
    selectedZoom: 17,
    selectMap: null,
    selectMarker: null,
    
    // Uploads
    customFiles: null,
    
    // Results & Swipe Stage
    resultsData: null,
    swipePosPct: 50,
    isDraggingSwipe: false,
    
    // Pan & Zoom on Stage
    zoomScale: 1.0,
    panX: 0,
    panY: 0,
    isPanning: false,
    panStartX: 0,
    panStartY: 0,
    allVectorsVisible: true,
    
    // Highlighted Building
    selectedBuildingId: null,

    // View Mode & Filters
    viewMode: localStorage.getItem('atlas_view_mode') || 'grid',
    activeFilterScope: 'all',
    activeSortBy: 'updated'
};

// DOM References
const el = {
    // Screen Views
    viewDashboard: document.getElementById('view-dashboard'),
    viewWorkspace: document.getElementById('view-workspace'),
    
    // Dashboard Sidebar & Header
    btnSidebarCreateProject: document.getElementById('btn-sidebar-create-project'),
    btnBannerNewProject: document.getElementById('btn-banner-new-project'),
    btnShortcutCreate: document.getElementById('btn-shortcut-create'),
    btnShortcutImport: document.getElementById('btn-shortcut-import'),
    btnShortcutTemplates: document.getElementById('btn-shortcut-templates'),
    inputDashSearch: document.getElementById('input-dash-search'),
    dashFilterScope: document.getElementById('dash-filter-scope'),
    dashSortBy: document.getElementById('dash-sort-by'),
    btnViewGrid: document.getElementById('btn-view-grid'),
    btnViewList: document.getElementById('btn-view-list'),
    containerRecentsProjects: document.getElementById('container-recents-projects'),
    containerAllProjects: document.getElementById('container-all-projects'),
    lblAllProjectsCount: document.getElementById('lbl-all-projects-count'),

    // Sidebar Menu
    menuBtnDiscover: document.getElementById('menu-btn-discover'),
    menuBtnProjects: document.getElementById('menu-btn-projects'),
    menuBtnSettings: document.getElementById('menu-btn-settings'),

    // Modals
    modalDiscover: document.getElementById('modal-discover'),
    modalSettings: document.getElementById('modal-settings'),

    // Settings Modal Controls
    rangeSettingMinarea: document.getElementById('range-setting-minarea'),
    lblSettingMinarea: document.getElementById('lbl-setting-minarea'),
    btnSaveSettings: document.getElementById('btn-save-settings'),
    btnClearAllProjects: document.getElementById('btn-clear-all-projects'),

    
    // Workspace Topbar
    btnBackToDashboard: document.getElementById('btn-back-to-dashboard'),
    inputTopbarProjectName: document.getElementById('input-topbar-project-name'),
    lblProjectSaveState: document.getElementById('lbl-project-save-state'),
    btnManualSaveProject: document.getElementById('btn-manual-save-project'),
    
    // Stepper Nav
    stepNavs: [
        document.getElementById('step-nav-1'),
        document.getElementById('step-nav-2'),
        document.getElementById('step-nav-3')
    ],
    views: [
        document.getElementById('view-step-1'),
        document.getElementById('view-step-2'),
        document.getElementById('view-step-3')
    ],
    
    // Step 1
    inputStep1ProjectName: document.getElementById('input-step1-project-name'),
    sourceCards: document.querySelectorAll('.source-card'),
    panels: {
        'live-hotspot': document.getElementById('panel-live-hotspot'),
        'map-click': document.getElementById('panel-map-click'),
        'custom-upload': document.getElementById('panel-custom-upload')
    },
    selectHotspot: document.getElementById('select-hotspot'),
    lblHotspotName: document.getElementById('lbl-hotspot-name'),
    lblHotspotDesc: document.getElementById('lbl-hotspot-desc'),
    yearT1: document.getElementById('year-t1'),
    yearT2: document.getElementById('year-t2'),
    mapYearT1: document.getElementById('map-year-t1'),
    mapYearT2: document.getElementById('map-year-t2'),
    lblSelectedCoords: document.getElementById('lbl-selected-coords'),
    lblSelectedZoom: document.getElementById('lbl-selected-zoom'),
    inputLiveArea: document.getElementById('input-live-area'),
    lblLiveAreaInfo: document.getElementById('lbl-live-area-info'),
    inputUploadT1: document.getElementById('input-upload-t1'),
    inputUploadT2: document.getElementById('input-upload-t2'),
    inputWorldT1: document.getElementById('input-world-t1'),
    inputWorldT2: document.getElementById('input-world-t2'),
    inputGeorefGsd: document.getElementById('input-georef-gsd'),
    inputGeorefEpsg: document.getElementById('input-georef-epsg'),
    inputGeorefSouth: document.getElementById('input-georef-south'),
    inputGeorefWest: document.getElementById('input-georef-west'),
    inputGeorefNorth: document.getElementById('input-georef-north'),
    inputGeorefEast: document.getElementById('input-georef-east'),
    btnGotoStep2: document.getElementById('btn-goto-step-2'),
    
    // Step 2
    step2InfoText: document.getElementById('step-2-info-text'),
    lblPreviewT1: document.getElementById('lbl-preview-t1'),
    lblPreviewT2: document.getElementById('lbl-preview-t2'),
    imgPreviewT1: document.getElementById('img-preview-t1'),
    imgPreviewT2: document.getElementById('img-preview-t2'),
    btnBackToStep1: document.getElementById('btn-back-to-step-1'),
    btnRunBuildingDetection: document.getElementById('btn-run-building-detection'),
    
    // Step 3
    btnRestartWizard: document.getElementById('btn-restart-wizard'),
    resCountNew: document.getElementById('res-count-new'),
    resCountRebuilt: document.getElementById('res-count-rebuilt'),
    resCountDem: document.getElementById('res-count-dem'),
    resCountExist: document.getElementById('res-count-exist'),
    resTotalArea: document.getElementById('res-total-area'),
    lblBuildingCount: document.getElementById('lbl-building-count'),
    buildingItemsList: document.getElementById('building-items-list'),
    
    // Swipe Bar & Mode
    badgeYearT1: document.getElementById('badge-year-t1'),
    badgeYearT2: document.getElementById('badge-year-t2'),
    lblSwipeY1: document.getElementById('lbl-swipe-y1'),
    lblSwipeY2: document.getElementById('lbl-swipe-y2'),
    lblSwipeY1Tag: document.getElementById('lbl-swipe-y1-tag'),
    lblSwipeY2Tag: document.getElementById('lbl-swipe-y2-tag'),
    btnTogglePolygons: document.getElementById('btn-toggle-polygons'),
    lblTogglePoly: document.getElementById('lbl-toggle-poly'),
    
    // Swipe Stage & Canvas Elements
    swipeStageOuter: document.getElementById('swipe-stage-outer'),
    swipeStageViewport: document.getElementById('swipe-stage-viewport'),
    swipeStageContent: document.getElementById('swipe-stage-content'),
    stageImgT2: document.getElementById('stage-img-t2'),
    stageImgT1: document.getElementById('stage-img-t1'),
    stageT1Clipper: document.getElementById('stage-t1-clipper'),
    stageVectorSvg: document.getElementById('stage-vector-svg'),
    svgGroupExist: document.getElementById('svg-group-exist'),
    svgGroupNew: document.getElementById('svg-group-new'),
    svgGroupRebuilt: document.getElementById('svg-group-rebuilt'),
    svgGroupDem: document.getElementById('svg-group-dem'),
    svgGroupLabels: document.getElementById('svg-group-labels'),
    swipeDividerLine: document.getElementById('swipe-divider-line'),
    swipeDividerHandle: document.getElementById('swipe-divider-handle'),
    buildingHoverTooltip: document.getElementById('building-hover-tooltip'),
    
    // Floating Vector Layer Checkboxes
    chkLayerNew: document.getElementById('chk-layer-new'),
    chkLayerRebuilt: document.getElementById('chk-layer-rebuilt'),
    chkLayerDem: document.getElementById('chk-layer-dem'),
    chkLayerExist: document.getElementById('chk-layer-exist'),
    chkLayerLabels: document.getElementById('chk-layer-labels'),
    layerBadgeNew: document.getElementById('layer-badge-new'),
    layerBadgeRebuilt: document.getElementById('layer-badge-rebuilt'),
    layerBadgeDem: document.getElementById('layer-badge-dem'),
    layerBadgeExist: document.getElementById('layer-badge-exist'),
    
    // Zoom & Reset Toolbar
    btnZoomIn: document.getElementById('btn-zoom-in'),
    btnZoomOut: document.getElementById('btn-zoom-out'),
    btnZoomReset: document.getElementById('btn-zoom-reset'),
    lblZoomLevel: document.getElementById('lbl-zoom-level'),
    
    // Corner tags
    swipeTagLeft: document.getElementById('swipe-tag-left'),
    swipeTagRight: document.getElementById('swipe-tag-right'),
    
    btnExportGeoJson: document.getElementById('btn-export-geojson'),
    btnExportCsv: document.getElementById('btn-export-csv')
};

// ==========================================
// SCREEN SWITCHING (Dashboard vs Workspace)
// ==========================================
function switchScreen(screen) {
    state.activeScreen = screen;
    if (screen === 'dashboard') {
        el.viewDashboard.classList.add('active');
        el.viewWorkspace.classList.remove('active');
        loadProjects();
    } else {
        el.viewWorkspace.classList.add('active');
        el.viewDashboard.classList.remove('active');
        setTimeout(() => {
            if (state.currentStep === 3) {
                updateSwipeDivider(state.swipePosPct);
            }
        }, 100);
    }
}

// Wizard Step Navigation inside Workspace
function goToStep(step) {
    state.currentStep = step;
    
    el.stepNavs.forEach((nav, idx) => {
        const stepNum = idx + 1;
        nav.classList.remove('active', 'completed');
        if (stepNum === step) {
            nav.classList.add('active');
        } else if (stepNum < step) {
            nav.classList.add('completed');
        }
    });
    
    el.views.forEach((view, idx) => {
        if (idx + 1 === step) {
            view.classList.add('active');
        } else {
            view.classList.remove('active');
        }
    });

    if (step === 3) {
        setTimeout(() => {
            updateSwipeDivider(state.swipePosPct);
        }, 100);
    }
}

// ==========================================
// PROJECTS PERSISTENCE & API
// ==========================================
async function loadProjects(searchQuery = '') {
    try {
        const q = searchQuery !== '' ? searchQuery : (el.inputDashSearch ? el.inputDashSearch.value : '');
        const res = await fetch(`/api/projects?search=${encodeURIComponent(q)}`);
        const data = await res.json();
        if (data.success) {
            state.projectsList = data.projects || [];
            applyFilterAndSort();
        }
    } catch (err) {
        console.error('Error loading projects:', err);
    }
}

function applyFilterAndSort() {
    let list = [...state.projectsList];

    // 1. Filter by Scope / Collection
    const scope = state.activeFilterScope;
    if (scope === 'recent') {
        list = list.slice(0, 5);
    } else if (scope === 'kentsel') {
        list = list.filter(p => {
            const str = ((p.name || '') + ' ' + (p.location_name || '') + ' ' + (p.collection || '')).toLowerCase();
            return str.includes('kentsel') || str.includes('dönüşüm') || str.includes('fikirtepe') || str.includes('kadıköy') || str.includes('mamak') || str.includes('örnekköy');
        });
    } else if (scope === 'konut') {
        list = list.filter(p => {
            const str = ((p.name || '') + ' ' + (p.location_name || '') + ' ' + (p.collection || '')).toLowerCase();
            return str.includes('konut') || str.includes('başakşehir') || str.includes('kayaşehir') || str.includes('incek') || str.includes('müstakil') || str.includes('teksas');
        });
    } else if (scope === 'sanayi') {
        list = list.filter(p => {
            const str = ((p.name || '') + ' ' + (p.location_name || '') + ' ' + (p.collection || '')).toLowerCase();
            return str.includes('sanayi') || str.includes('austin') || str.includes('giga') || str.includes('dubai') || str.includes('fabrika') || str.includes('lojistik');
        });
    }

    // 2. Sort
    const sort = state.activeSortBy;
    if (sort === 'name_asc') {
        list.sort((a, b) => (a.name || '').localeCompare(b.name || '', 'tr'));
    } else if (sort === 'area_desc') {
        list.sort((a, b) => ((b.stats?.total_changed_m2 || 0) - (a.stats?.total_changed_m2 || 0)));
    } else if (sort === 'created') {
        list.sort((a, b) => (new Date(b.created_at || 0) - new Date(a.created_at || 0)));
    } else {
        // 'updated'
        list.sort((a, b) => (new Date(b.updated_at || b.created_at || 0) - new Date(a.updated_at || a.created_at || 0)));
    }

    renderProjectsList(list);
}

function setViewMode(mode) {
    state.viewMode = mode;
    if (mode === 'list') {
        if (el.btnViewList) el.btnViewList.classList.add('active');
        if (el.btnViewGrid) el.btnViewGrid.classList.remove('active');
        if (el.containerRecentsProjects) el.containerRecentsProjects.classList.add('view-list-mode');
        if (el.containerAllProjects) el.containerAllProjects.classList.add('view-list-mode');
    } else {
        if (el.btnViewGrid) el.btnViewGrid.classList.add('active');
        if (el.btnViewList) el.btnViewList.classList.remove('active');
        if (el.containerRecentsProjects) el.containerRecentsProjects.classList.remove('view-list-mode');
        if (el.containerAllProjects) el.containerAllProjects.classList.remove('view-list-mode');
    }
    try {
        localStorage.setItem('atlas_view_mode', mode);
    } catch(e) {}
}

function renderProjectsList(projects) {
    if (el.lblAllProjectsCount) {
        el.lblAllProjectsCount.textContent = `${projects.length} proje`;
    }

    if (el.containerRecentsProjects) el.containerRecentsProjects.innerHTML = '';
    if (el.containerAllProjects) el.containerAllProjects.innerHTML = '';

    if (!projects || projects.length === 0) {
        const emptyHtml = `
            <div style="grid-column: 1 / -1; padding: 40px; text-align: center; background: #ffffff; border: 1px dashed #cbd5e1; border-radius: 12px; color: #64748b;">
                <i class="fa-solid fa-folder-open" style="font-size: 2rem; color: #94a3b8; margin-bottom: 10px;"></i>
                <h4 style="font-weight: 700; color: #1e293b; margin-bottom: 4px;">Kriterlere Uygun Proje Bulunamadı</h4>
                <p style="font-size: 0.82rem; margin: 0;">Filtre veya arama kelimesini değiştirin veya yukarıdaki butonla yeni proje oluşturun.</p>
            </div>
        `;
        if (el.containerRecentsProjects) el.containerRecentsProjects.innerHTML = emptyHtml;
        if (el.containerAllProjects) el.containerAllProjects.innerHTML = emptyHtml;
        return;
    }

    // Recents: Top 3
    const recents = projects.slice(0, 3);
    recents.forEach(proj => {
        const card = createProjectCardElement(proj);
        if (el.containerRecentsProjects) el.containerRecentsProjects.appendChild(card);
    });

    // All Projects
    projects.forEach(proj => {
        const card = createProjectCardElement(proj);
        if (el.containerAllProjects) el.containerAllProjects.appendChild(card);
    });

    // Apply view mode styling
    if (state.viewMode === 'list') {
        if (el.containerRecentsProjects) el.containerRecentsProjects.classList.add('view-list-mode');
        if (el.containerAllProjects) el.containerAllProjects.classList.add('view-list-mode');
    } else {
        if (el.containerRecentsProjects) el.containerRecentsProjects.classList.remove('view-list-mode');
        if (el.containerAllProjects) el.containerAllProjects.classList.remove('view-list-mode');
    }
}

function createProjectCardElement(proj) {
    const card = document.createElement('div');
    card.className = 'project-card';
    card.setAttribute('data-id', proj.id);

    // Thumbnail: use project thumbnail_url or fallback
    const thumbSrc = proj.thumbnail_url || "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='320' height='180'><rect fill='%230f172a' width='320' height='180'/><text fill='%2338bdf8' x='50%' y='50%' dominant-baseline='middle' text-anchor='middle' font-family='sans-serif' font-weight='bold'>Uydu Analizi</text></svg>";
    
    const stats = proj.stats || {};
    const newCount = stats.new_buildings_count ?? 0;
    const demCount = stats.demolished_count ?? 0;
    const totalM2 = stats.total_changed_m2 ?? 0;
    const isShared = proj.is_shared || false;
    const scopeBadge = isShared 
        ? `<span class="card-badge-scope"><i class="fa-solid fa-users"></i> Shared</span>` 
        : `<span class="card-badge-scope"><i class="fa-solid fa-lock"></i> Private</span>`;

    card.innerHTML = `
        <div class="project-card-thumb">
            <img src="${thumbSrc}" alt="${proj.name}">
            ${scopeBadge}
            <div class="card-dots-btn" title="Projeyi Sil" data-action="delete">
                <i class="fa-solid fa-trash-can"></i>
            </div>
        </div>
        <div class="project-card-body">
            <h3 class="project-card-title">${proj.name || 'İsimsiz Proje'}</h3>
            <div class="project-card-meta">
                <i class="fa-regular fa-calendar"></i> ${proj.created_at_formatted || 'Yeni'}
                <span>•</span>
                <span>${proj.location_name || 'Uydu Bölgesi'}</span>
            </div>
            <div class="project-card-stats-row">
                <span class="stat-tag-mini stat-tag-new">🟢 +${newCount} Yeni</span>
                <span class="stat-tag-mini stat-tag-dem">🔴 -${demCount} Yıkılan</span>
                ${totalM2 > 0 ? `<span class="stat-tag-mini stat-tag-exist">📐 ${totalM2} m²</span>` : ''}
            </div>
        </div>
    `;

    // Click on card: open project
    card.addEventListener('click', (e) => {
        if (e.target.closest('[data-action="delete"]')) return;
        openProject(proj.id);
    });

    // Click on delete
    const deleteBtn = card.querySelector('[data-action="delete"]');
    if (deleteBtn) {
        deleteBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            deleteProject(proj.id, proj.name);
        });
    }

    return card;
}

// Open existing project from database
async function openProject(projectId) {
    try {
        const res = await fetch(`/api/projects/${projectId}`);
        const data = await res.json();
        if (!data.success || !data.project) {
            showToast('Proje açılamadı: ' + (data.error || ''), 'error');
            return;
        }

        const proj = data.project;
        state.currentProjectId = proj.id;
        state.currentProjectName = proj.name || 'Proje';
        state.sourceType = proj.source_type || 'live-hotspot';
        state.selectedCoords = proj.coords || [41.1070, 28.7900];
        state.selectedZoom = proj.zoom || 17;
        
        el.inputTopbarProjectName.value = state.currentProjectName;
        if (el.inputStep1ProjectName) el.inputStep1ProjectName.value = state.currentProjectName;
        setSaveIndicator(true);

        switchScreen('workspace');

        if (proj.results_data && proj.results_data.overlays) {
            state.resultsData = proj.results_data;
            const y1 = proj.year_t1 || 'T1';
            const y2 = proj.year_t2 || 'T2';
            el.lblPreviewT1.textContent = y1;
            el.lblPreviewT2.textContent = y2;
            el.badgeYearT1.innerHTML = `<i class="fa-solid fa-backward"></i> Solda 1. Görüntü: ${y1}`;
            el.badgeYearT2.innerHTML = `<i class="fa-solid fa-forward"></i> Sağda 2. Görüntü: ${y2}`;
            el.lblSwipeY1.textContent = y1;
            el.lblSwipeY2.textContent = y2;

            goToStep(3);
            setTimeout(() => {
                renderPureImageResults(state.resultsData);
            }, 100);
        } else {
            goToStep(1);
        }
    } catch (err) {
        console.error('Error opening project:', err);
        showToast('Proje yüklenirken bir sorun oluştu.', 'error');
    }
}

// Delete project
async function deleteProject(projectId, projectName) {
    const confirmed = await showConfirm({
        title: 'Projeyi Sil',
        message: `<strong>"${projectName}"</strong> projesini kalıcı olarak silmek istediğinize emin misiniz?`,
        icon: 'danger',
        confirmText: 'Evet, Sil',
        cancelText: 'Vazgeç'
    });
    if (!confirmed) return;

    try {
        const res = await fetch(`/api/projects/${projectId}`, { method: 'DELETE' });
        const data = await res.json();
        if (data.success) {
            showToast('Proje başarıyla silindi.', 'success');
            loadProjects();
        } else {
            showToast('Proje silinemedi: ' + (data.error || ''), 'error');
        }
    } catch (err) {
        console.error('Error deleting project:', err);
    }
}

// Save project to database
async function saveProjectToDatabase(showNotice = false) {
    const projName = el.inputTopbarProjectName.value.trim() || el.inputStep1ProjectName.value.trim() || 'İsimsiz Analiz Projesi';
    state.currentProjectName = projName;

    const payload = {
        id: state.currentProjectId || undefined,
        name: projName,
        source_type: state.sourceType,
        location_name: getLocationTitle(),
        year_t1: (state.sourceType === 'map-click') ? el.mapYearT1?.value : el.yearT1?.value,
        year_t2: (state.sourceType === 'map-click') ? el.mapYearT2?.value : el.yearT2?.value,
        coords: state.selectedCoords,
        zoom: state.selectedZoom,
        stats: state.resultsData?.stats || {},
        thumbnail_url: state.resultsData?.overlays?.t2_png_base64 || '',
        results_data: state.resultsData || null
    };

    try {
        const res = await fetch('/api/projects', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.success && data.project) {
            state.currentProjectId = data.project.id;
            setSaveIndicator(true);
            if (showNotice) {
                showToast('Proje başarıyla kaydedildi!', 'success');
            }
        }
    } catch (err) {
        console.error('Error saving project:', err);
    }
}

function setSaveIndicator(isSaved) {
    if (!el.lblProjectSaveState) return;
    if (isSaved) {
        el.lblProjectSaveState.innerHTML = '<i class="fa-solid fa-circle-check"></i> Kaydedildi';
        el.lblProjectSaveState.style.color = '#34d399';
    } else {
        el.lblProjectSaveState.innerHTML = '<i class="fa-solid fa-circle-dot"></i> Kaydedilmedi';
        el.lblProjectSaveState.style.color = '#f59e0b';
    }
}

function getLocationTitle() {
    if (state.sourceType === 'live-hotspot') {
        const h = state.hotspots[state.currentHotspotId];
        return h ? h.title.split('(')[0].trim() : 'Canlı Uydu Bölgesi';
    } else if (state.sourceType === 'map-click') {
        return `Harita [${state.selectedCoords[0].toFixed(2)}, ${state.selectedCoords[1].toFixed(2)}]`;
    }
    return 'Özel Yükleme';
}

function createNewProject(prefillSource = null) {
    state.currentProjectId = null;
    state.currentProjectName = 'Yeni Bina Değişim Projesi';
    state.resultsData = null;
    
    el.inputTopbarProjectName.value = state.currentProjectName;
    if (el.inputStep1ProjectName) el.inputStep1ProjectName.value = state.currentProjectName;
    setSaveIndicator(false);

    if (prefillSource) {
        const card = document.querySelector(`.source-card[data-source="${prefillSource}"]`);
        if (card) card.click();
    }

    switchScreen('workspace');
    goToStep(1);
}

// ==========================================
// HOTSPOTS & STEP 1 INITIALIZATION
// ==========================================
async function loadHotspots() {
    try {
        const res = await fetch('/api/live/hotspots');
        const data = await res.json();
        el.selectHotspot.innerHTML = '';
        data.forEach(h => {
            state.hotspots[h.id] = h;
            const opt = document.createElement('option');
            opt.value = h.id;
            opt.textContent = h.title;
            el.selectHotspot.appendChild(opt);
        });
        if (data.length > 0) {
            updateHotspotCard(data[0].id);
        }
    } catch (err) {
        console.error('Error loading hotspots:', err);
    }
}

function updateHotspotCard(hotspotId) {
    const h = state.hotspots[hotspotId];
    if (!h) return;
    state.currentHotspotId = hotspotId;
    state.selectedCoords = [h.lat, h.lon];
    state.selectedZoom = h.zoom || 17;
    el.lblHotspotName.textContent = h.title;
    el.lblHotspotDesc.textContent = h.desc;
    if (h.year_t1) el.yearT1.value = h.year_t1;
    if (h.year_t2) el.yearT2.value = h.year_t2;

    // Suggest project name if new project
    if (!state.currentProjectId && el.inputStep1ProjectName) {
        const defaultName = `${h.title.split('(')[0].replace(/[🇹🇷🇺🇸🇦🇪]/g, '').trim()} Kentsel Değişim`;
        el.inputStep1ProjectName.value = defaultName;
        el.inputTopbarProjectName.value = defaultName;
    }
}

// Live Wayback patches are a grid of zoom-17 tiles centred on the point
// (same maths as geo.centred_grid_origin on the server)
const LIVE_ZOOM = 17;
const MAX_LIVE_GRID = 8;

function liveTileSizeM(lat) {
    return 256 * 156543.03392 * Math.cos(lat * Math.PI / 180) / Math.pow(2, LIVE_ZOOM);
}

// Tiles per side for the side length entered in metres
function liveGridSize() {
    const lat = state.selectedCoords[0] || 41;
    const side = parseFloat(el.inputLiveArea?.value) || 460;
    return Math.min(MAX_LIVE_GRID, Math.max(1, Math.round(side / liveTileSizeM(lat))));
}

function liveGridTiles(lat, lon, n) {
    const z2 = Math.pow(2, LIVE_ZOOM);
    const xf = (lon + 180) / 360 * z2;
    const yf = (1 - Math.asinh(Math.tan(lat * Math.PI / 180)) / Math.PI) / 2 * z2;
    return {
        x0: Math.floor(xf - n / 2 + 0.5),
        y0: Math.floor(yf - n / 2 + 0.5),
        lonOf: x => x / z2 * 360 - 180,
        latOf: y => Math.atan(Math.sinh(Math.PI * (1 - 2 * y / z2))) * 180 / Math.PI
    };
}

// Draws the exact area (and its tile grid) that the analysis will download around the pin
function drawLiveArea(fit = false) {
    if (!state.selectMap) return;
    const [lat, lon] = state.selectedCoords;
    const n = liveGridSize();
    const { x0, y0, lonOf, latOf } = liveGridTiles(lat, lon, n);
    const south = latOf(y0 + n), north = latOf(y0), west = lonOf(x0), east = lonOf(x0 + n);
    if (state.liveAreaLayer) state.liveAreaLayer.remove();
    const layer = L.layerGroup();
    L.rectangle([[south, west], [north, east]], { color: '#f59e0b', weight: 3, fillOpacity: 0.12, interactive: false }).addTo(layer);
    const lineStyle = { color: '#fbbf24', weight: 2, opacity: 1, dashArray: '6 4', interactive: false };
    for (let i = 1; i < n; i++) {
        L.polyline([[latOf(y0 + i), west], [latOf(y0 + i), east]], lineStyle).addTo(layer);
        L.polyline([[south, lonOf(x0 + i)], [north, lonOf(x0 + i)]], lineStyle).addTo(layer);
    }
    layer.addTo(state.selectMap);
    state.liveAreaLayer = layer;
    const sideM = Math.round(n * liveTileSizeM(lat));
    if (el.lblLiveAreaInfo) {
        el.lblLiveAreaInfo.textContent = `${n}×${n} karo · ~${sideM} m × ${sideM} m · T1+T2 için ${2 * n * n} karo indirilecek`;
    }
    if (fit) state.selectMap.fitBounds([[south, west], [north, east]], { padding: [30, 30], maxZoom: 17 });
}

// Interactive Map Picker Initialization (Step 1) with Esri Satellite Basemap
function initSelectMap() {
    if (state.selectMap) return;
    const defaultLat = state.selectedCoords[0] || 41.1070;
    const defaultLon = state.selectedCoords[1] || 28.7900;
    
    state.selectMap = L.map('select-map', {
        center: [defaultLat, defaultLon],
        zoom: 14,
        zoomControl: true
    });

    // 1. Esri World Imagery (Satellite) Basemap
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        maxZoom: 19,
        attribution: '&copy; Esri, Maxar, Earthstar Geographics'
    }).addTo(state.selectMap);

    // 2. Esri Boundaries & Places Labels
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}', {
        maxZoom: 19,
        opacity: 0.85
    }).addTo(state.selectMap);

    // Red draggable marker
    state.selectMarker = L.marker([defaultLat, defaultLon], {
        draggable: true
    }).addTo(state.selectMap);

    const updateCoordDisplay = (lat, lon, zoom) => {
        state.selectedCoords = [parseFloat(lat.toFixed(5)), parseFloat(lon.toFixed(5))];
        state.selectedZoom = zoom || state.selectMap.getZoom();
        if (el.lblSelectedCoords) {
            el.lblSelectedCoords.textContent = `${state.selectedCoords[0]}, ${state.selectedCoords[1]}`;
        }
        if (el.lblSelectedZoom) {
            el.lblSelectedZoom.textContent = state.selectedZoom;
        }
        drawLiveArea();
        if (!state.currentProjectId && el.inputStep1ProjectName && state.sourceType === 'map-click') {
            const coordName = `Uydu Analizi [${state.selectedCoords[0]}, ${state.selectedCoords[1]}]`;
            el.inputStep1ProjectName.value = coordName;
            el.inputTopbarProjectName.value = coordName;
        }
    };

    state.selectMarker.on('dragend', (e) => {
        const p = e.target.getLatLng();
        updateCoordDisplay(p.lat, p.lng);
    });

    state.selectMap.on('click', (e) => {
        const lat = e.latlng.lat;
        const lon = e.latlng.lng;
        state.selectMarker.setLatLng([lat, lon]);
        updateCoordDisplay(lat, lon);
    });

    drawLiveArea();
    if (el.inputLiveArea) {
        el.inputLiveArea.addEventListener('change', () => drawLiveArea(true));
    }

    state.selectMap.on('zoomend', () => {
        if (el.lblSelectedZoom) {
            el.lblSelectedZoom.textContent = state.selectMap.getZoom();
        }
    });

    // Quick City Buttons
    document.querySelectorAll('.quick-city-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const lat = parseFloat(btn.getAttribute('data-lat'));
            const lon = parseFloat(btn.getAttribute('data-lon'));
            const zoom = parseInt(btn.getAttribute('data-zoom')) || 17;
            state.selectMap.flyTo([lat, lon], zoom);
            state.selectMarker.setLatLng([lat, lon]);
            updateCoordDisplay(lat, lon, zoom);
        });
    });
}

// ==========================================
// STEP 1 -> STEP 2: Prepare or Fetch Images
// ==========================================
// ==========================================
// GEOREFERENCE, ANALYSIS MODE & DETECTION JOBS
// ==========================================
const MAX_EXISTING_CARDS = 200;      // sidebar cards for unchanged buildings (changed ones are always listed)
const MAX_LABELLED_BUILDINGS = 300;  // above this, only changed buildings get SVG number labels

// World files and .prj share one multi-file input per image; the server expects world_* / prj_* fields
function appendSidecars(formData, input, suffix) {
    Array.from(input?.files || []).forEach(file => {
        const field = file.name.toLowerCase().endsWith('.prj') ? `prj_${suffix}` : `world_${suffix}`;
        formData.append(field, file);
    });
}

function getManualGeoref() {
    const num = (input) => (input && input.value !== '' ? parseFloat(input.value) : null);
    const manual = {};
    const gsd = num(el.inputGeorefGsd);
    const epsg = num(el.inputGeorefEpsg);
    const bounds = [el.inputGeorefSouth, el.inputGeorefWest, el.inputGeorefNorth, el.inputGeorefEast].map(num);
    if (gsd !== null) manual.gsd = gsd;
    if (epsg !== null) manual.epsg = epsg;
    if (bounds.every(v => v !== null)) manual.bounds = bounds;
    return manual;
}

function getAnalysisMode() {
    const checked = document.querySelector('input[name="analysis-mode"]:checked');
    return checked ? checked.value : 'fast';
}

function describeGeoref(label, georef, error) {
    if (error) return `${label}: konum okunamadı (${error})`;
    if (!georef) return `${label}: georeferanssız`;
    const source = georef.source === 'geotiff' ? 'GeoTIFF' : 'World file';
    return `${label}: ${source} · ${georef.crs} · ${georef.gsd_m} m/piksel`;
}

function formatArea(areaM2, areaPx) {
    if (areaM2 !== null && areaM2 !== undefined) return `${areaM2} m²`;
    if (areaPx !== null && areaPx !== undefined) return `${areaPx} piksel`;
    return '—';
}

// POST /api/detect; engine "ml" answers 202 + job_id, so poll /api/jobs until the job ends
// A Wayback "year" is the archive release; the imagery in it can be years older
function yearWithCapture(year, capture) {
    return capture && capture.date ? `${year} · çekim ${capture.date.slice(0, 7)}` : year;
}

function showCaptureInfo(capture, y1, y2) {
    const c = capture || {};
    el.lblPreviewT1.textContent = yearWithCapture(y1, c.t1);
    el.lblPreviewT2.textContent = yearWithCapture(y2, c.t2);
    if (c.t1 && c.t2) {
        const describe = (k, x) => `${k} ${x.date}${x.source ? ' (' + x.source + ')' : ''}`;
        el.step2InfoText.textContent += ` Gerçek çekim tarihleri: ${describe('T1', c.t1)}, ${describe('T2', c.t2)}.`;
        if (c.t1.date === c.t2.date) {
            showToast(`Seçilen iki arşiv yılı aynı çekimi gösteriyor (${c.t1.date}); değişim bulunamaz, farklı yıllar seçin.`, 'warning', 9000);
        }
    }
}

// Shows a step-2 preview with the dashed tile grid the area was split into
function showWithTileGrid(imgEl, src, n) {
    if (!n || n <= 1) { imgEl.src = src; return; }
    const img = new Image();
    img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
        ctx.strokeStyle = 'rgba(245, 158, 11, 0.9)';
        ctx.lineWidth = Math.max(1, img.width / 400);
        ctx.setLineDash([8, 6]);
        for (let i = 1; i < n; i++) {
            const x = img.width * i / n;
            const y = img.height * i / n;
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, img.height); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(img.width, y); ctx.stroke();
        }
        imgEl.src = canvas.toDataURL('image/jpeg', 0.9);
    };
    img.src = src;
}

async function runDetectionJob(payload, url = '/api/detect') {
    const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    const start = await res.json();
    if (!start.success || !start.job_id) return start;
    for (;;) {
        await new Promise(resolve => setTimeout(resolve, 1500));
        const job = await (await fetch(`/api/jobs/${start.job_id}`)).json();
        if (!job.success) return job;
        if (job.status === 'done') return job.result;
        if (job.status === 'error') return { success: false, error: job.error };
        const pct = Math.round((job.progress || 0) * 100);
        el.btnRunBuildingDetection.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> %${pct} · ${job.stage}`;
    }
}

async function handleStep1Next() {
    el.btnGotoStep2.disabled = true;
    el.btnGotoStep2.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Görüntüler Hazırlanıyor...';

    const isMapClick = (state.sourceType === 'map-click');
    const y1 = isMapClick ? (el.mapYearT1?.value || '2014') : el.yearT1.value;
    const y2 = isMapClick ? (el.mapYearT2?.value || '2026') : el.yearT2.value;
    
    el.lblPreviewT1.textContent = y1;
    el.lblPreviewT2.textContent = y2;
    el.badgeYearT1.innerHTML = `<i class="fa-solid fa-backward"></i> Solda 1. Görüntü: ${y1}`;
    el.badgeYearT2.innerHTML = `<i class="fa-solid fa-forward"></i> Sağda 2. Görüntü: ${y2}`;
    el.lblSwipeY1.textContent = y1;
    el.lblSwipeY2.textContent = y2;

    try {
        if (state.sourceType === 'live-hotspot' || state.sourceType === 'map-click') {
            el.step2InfoText.textContent = isMapClick
                ? `Haritadan seçilen koordinatın (${state.selectedCoords[0]}, ${state.selectedCoords[1]}) ${y1} ve ${y2} canlı uydu fotoğrafları getirildi.`
                : `Seçilen bölgenin ${y1} ve ${y2} canlı uydu fotoğrafları getirildi.`;
            
            el.imgPreviewT1.src = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='256' height='256'><rect fill='%231e293b' width='256' height='256'/><text fill='%2364748b' x='50%' y='50%' dominant-baseline='middle' text-anchor='middle'>Uydu Çekiliyor...</text></svg>";
            el.imgPreviewT2.src = el.imgPreviewT1.src;
            
            goToStep(2);

            // Step 2 only needs the imagery (1-2 s); building detection runs on "Binaları Bul"
            state.resultsData = null;
            state.livePayload = {
                lat: state.selectedCoords[0],
                lon: state.selectedCoords[1],
                year_t1: y1,
                year_t2: y2,
                grid_size: isMapClick ? liveGridSize() : (state.hotspots[state.currentHotspotId]?.grid_size || 2)
            };
            const res = await fetch('/api/live/preview', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(state.livePayload)
            });
            const data = await res.json();
            if (data.success) {
                showWithTileGrid(el.imgPreviewT1, data.t1, data.grid_size);
                showWithTileGrid(el.imgPreviewT2, data.t2, data.grid_size);
                const n = data.grid_size;
                const sideM = Math.round(n * liveTileSizeM(state.selectedCoords[0]));
                el.step2InfoText.textContent += ` Alan ${n}×${n} karoya bölündü (~${sideM} m × ${sideM} m).`;
                showCaptureInfo(data.capture, y1, y2);
                const m = data.missing_tiles;
                if (m && (m.t1 || m.t2)) {
                    showToast(`Arşivde eksik karo var (T1: ${m.t1}, T2: ${m.t2} / ${m.total}); bu alanlar analiz dışı bırakılacak.`, 'warning', 9000);
                }
            } else {
                showToast('Uydu verisi getirilemedi: ' + (data.error || ''), 'error');
            }

        } else if (state.sourceType === 'custom-upload') {
            if (!el.inputUploadT1.files[0] || !el.inputUploadT2.files[0]) {
                showToast('Lütfen hem Zaman 1 (T1) hem de Zaman 2 (T2) fotoğraflarını seçin.', 'warning');
                el.btnGotoStep2.disabled = false;
                el.btnGotoStep2.innerHTML = 'Görüntüleri Getir ve 2. Adıma Geç <i class="fa-solid fa-arrow-right"></i>';
                return;
            }
            
            const formData = new FormData();
            formData.append('image_t1', el.inputUploadT1.files[0]);
            formData.append('image_t2', el.inputUploadT2.files[0]);
            appendSidecars(formData, el.inputWorldT1, 't1');
            appendSidecars(formData, el.inputWorldT2, 't2');
            
            const uploadRes = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            const uploadData = await uploadRes.json();
            if (!uploadData.success) {
                showToast('Yükleme hatası: ' + uploadData.error, 'error');
                return;
            }
            state.customFiles = uploadData;
            el.imgPreviewT1.src = uploadData.url_A;
            el.imgPreviewT2.src = uploadData.url_B;
            el.lblPreviewT1.textContent = "Yüklenen T1";
            el.lblPreviewT2.textContent = "Yüklenen T2";
            el.lblSwipeY1.textContent = "T1 (Önce)";
            el.lblSwipeY2.textContent = "T2 (Sonra)";
            el.badgeYearT1.innerHTML = `<i class="fa-solid fa-backward"></i> Solda 1. Görüntü: T1`;
            el.badgeYearT2.innerHTML = `<i class="fa-solid fa-forward"></i> Sağda 2. Görüntü: T2`;
            el.step2InfoText.textContent = `Yüklenen fotoğraflar hazırlandı. ${describeGeoref('T1', uploadData.georef_A, uploadData.georef_A_error)} — ${describeGeoref('T2', uploadData.georef_B, uploadData.georef_B_error)}`;
            state.resultsData = null;
            goToStep(2);
        }

    } catch (err) {
        console.error('Error in step 1 next:', err);
        showToast('Görüntüler hazırlanırken bir hata oluştu.', 'error');
    } finally {
        el.btnGotoStep2.disabled = false;
        el.btnGotoStep2.innerHTML = 'Görüntüleri Getir ve 2. Adıma Geç <i class="fa-solid fa-arrow-right"></i>';
    }
}

// ==========================================
// STEP 2 -> STEP 3: Run Building Detection
// ==========================================
async function handleRunDetection() {
    el.btnRunBuildingDetection.disabled = true;
    el.btnRunBuildingDetection.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Binalar Tespit Ediliyor...';

    try {
        let data = null;
        const aiConf = getAiSettings();

        if (state.sourceType === 'live-hotspot' || state.sourceType === 'map-click') {
            data = await runDetectionJob(
                { ...state.livePayload, min_area_m2: aiConf.minArea, engine: 'ml', job: true },
                '/api/live/detect'
            );
            state.resultsData = data;
        } else {
            if (state.sourceType === 'custom-upload') {
                data = await runDetectionJob({
                    engine: 'ml',
                    scenario_id: 'custom',
                    path_A: state.customFiles.path_A,
                    path_B: state.customFiles.path_B,
                    min_area_m2: aiConf.minArea,
                    analysis_mode: getAnalysisMode(),
                    manual_georef: getManualGeoref()
                });
            }
            state.resultsData = data;
        }

        if (!data || !data.success) {
            showToast('Bina tespiti yapılamadı: ' + (data ? data.error : ''), 'error');
            return;
        }

        // 1. Switch to Step 3
        goToStep(3);
        (data.warnings || []).forEach(w => showToast(w, 'warning', 9000));
        if (data.engine === 'ml' && !data.georef && state.sourceType === 'custom-upload') {
            showToast('Görüntüler georeferanssız: alanlar ve dışa aktarılan koordinatlar piksel cinsindendir.', 'info', 7000);
        }

        // 2. Render purely on downloaded/uploaded images in L.CRS.Simple
        setTimeout(() => {
            renderPureImageResults(data);
            // Automatically persist this analysis to projects database!
            saveProjectToDatabase(false);
        }, 150);

    } catch (err) {
        console.error('Detection error:', err);
        showToast('Bina değişim analizi sırasında hata oluştu.', 'error');
    } finally {
        el.btnRunBuildingDetection.disabled = false;
        el.btnRunBuildingDetection.innerHTML = '<i class="fa-solid fa-bolt"></i> Binaları Bul ve Değişimi Tespit Et <i class="fa-solid fa-arrow-right"></i>';
    }
}

// ==========================================
// STEP 3: RENDER PURE IMAGE RESULTS (SWIPE)
// ==========================================
function renderPureImageResults(data) {
    state.resultsData = data;
    const stats = data.stats || {};
    if (el.resCountNew) el.resCountNew.textContent = stats.new_buildings_count || 0;
    if (el.resCountDem) el.resCountDem.textContent = stats.demolished_count || 0;
    if (el.resCountExist) el.resCountExist.textContent = stats.existing_count || 0;
    if (el.resCountRebuilt) el.resCountRebuilt.textContent = stats.rebuilt_count || 0;
    if (el.resTotalArea) el.resTotalArea.textContent = formatArea(stats.total_changed_m2, stats.total_changed_px);
    
    // Layer badges
    if (el.layerBadgeNew) el.layerBadgeNew.textContent = stats.new_buildings_count || 0;
    if (el.layerBadgeDem) el.layerBadgeDem.textContent = stats.demolished_count || 0;
    if (el.layerBadgeExist) el.layerBadgeExist.textContent = stats.existing_count || 0;
    if (el.layerBadgeRebuilt) el.layerBadgeRebuilt.textContent = stats.rebuilt_count || 0;
    
    // Set years
    const capture = data.capture || {};
    const y1 = data.years ? yearWithCapture(data.years.t1, capture.t1) : 'T1';
    const y2 = data.years ? yearWithCapture(data.years.t2, capture.t2) : 'T2';
    if (el.lblSwipeY1) el.lblSwipeY1.textContent = y1;
    if (el.lblSwipeY2) el.lblSwipeY2.textContent = y2;
    if (el.lblSwipeY1Tag) el.lblSwipeY1Tag.textContent = y1;
    if (el.lblSwipeY2Tag) el.lblSwipeY2Tag.textContent = y2;
    
    // 1. Set Image Sources (T2 Base underneath, T1 Clipped Overlay on top)
    const overlays = data.overlays || {};
    if (overlays.t2_png_base64 && el.stageImgT2) {
        el.stageImgT2.src = overlays.t2_png_base64;
    }
    if (overlays.t1_png_base64 && el.stageImgT1) {
        el.stageImgT1.src = overlays.t1_png_base64;
    }

    // 2. Set SVG Canvas ViewBox
    const imgSize = data.image_size || [512, 512];
    const w = imgSize[0];
    const h = imgSize[1];
    if (el.stageVectorSvg) {
        el.stageVectorSvg.setAttribute('viewBox', `0 0 ${w} ${h}`);
    }

    // 3. Populate SVG Polygons and Labels
    if (el.svgGroupExist) el.svgGroupExist.innerHTML = '';
    if (el.svgGroupNew) el.svgGroupNew.innerHTML = '';
    if (el.svgGroupRebuilt) el.svgGroupRebuilt.innerHTML = '';
    if (el.svgGroupDem) el.svgGroupDem.innerHTML = '';
    if (el.svgGroupLabels) el.svgGroupLabels.innerHTML = '';

    const buildings = data.buildings || [];
    if (el.lblBuildingCount) el.lblBuildingCount.textContent = buildings.length;

    buildings.forEach(b => {
        if (!b.px_coords || b.px_coords.length < 3) return;

        const ptsStr = b.px_coords.map(pt => `${pt[0]},${pt[1]}`).join(' ');
        const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        polygon.setAttribute('points', ptsStr);
        polygon.setAttribute('data-id', b.id);
        polygon.id = `svg-poly-${b.id}`;

        const isNew = b.type === 'new';
        const isDem = b.type === 'demolished';
        const isRebuilt = b.type === 'rebuilt';
        const color = isNew ? '#10b981' : (isDem ? '#ef4444' : (isRebuilt ? '#f59e0b' : '#64748b'));
        const strokeColor = isNew ? '#059669' : (isDem ? '#dc2626' : (isRebuilt ? '#d97706' : '#475569'));

        polygon.setAttribute('fill', color);
        polygon.setAttribute('fill-opacity', '0.45');
        polygon.setAttribute('stroke', strokeColor);
        polygon.setAttribute('stroke-width', '2');

        // Polygon hover & click events
        polygon.addEventListener('mouseenter', (e) => {
            showBuildingTooltip(b, e);
            polygon.setAttribute('fill-opacity', '0.8');
            polygon.setAttribute('stroke-width', '3.5');
        });

        polygon.addEventListener('mouseleave', () => {
            hideBuildingTooltip();
            if (state.selectedBuildingId !== b.id) {
                polygon.setAttribute('fill-opacity', '0.45');
                polygon.setAttribute('stroke-width', '2');
            }
        });

        polygon.addEventListener('click', (e) => {
            e.stopPropagation();
            focusOnBuilding(b);
        });

        if (isNew && el.svgGroupNew) {
            el.svgGroupNew.appendChild(polygon);
        } else if (isDem && el.svgGroupDem) {
            el.svgGroupDem.appendChild(polygon);
        } else if (isRebuilt && el.svgGroupRebuilt) {
            el.svgGroupRebuilt.appendChild(polygon);
        } else if (el.svgGroupExist) {
            el.svgGroupExist.appendChild(polygon);
        }

        // Add Number Badge in SVG
        if (b.centroid_px && el.svgGroupLabels && (b.type !== 'existing' || buildings.length <= MAX_LABELLED_BUILDINGS)) {
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', b.centroid_px[0]);
            text.setAttribute('y', b.centroid_px[1]);
            text.setAttribute('text-anchor', 'middle');
            text.setAttribute('dominant-baseline', 'central');
            text.id = `svg-label-${b.id}`;
            text.textContent = `#${b.id}`;
            el.svgGroupLabels.appendChild(text);
        }
    });

    // 4. Populate Left Sidebar Building Cards
    if (el.buildingItemsList) {
        el.buildingItemsList.innerHTML = '';
        const changed = buildings.filter(b => b.type !== 'existing');
        const existing = buildings.filter(b => b.type === 'existing');
        const cardBuildings = changed.concat(existing.slice(0, MAX_EXISTING_CARDS));
        cardBuildings.forEach(b => {
            const item = document.createElement('div');
            item.className = 'building-card-item';
            item.id = `card-building-${b.id}`;

            let badgeClass = 'badge-b-exist';
            if (b.type === 'new') badgeClass = 'badge-b-new';
            else if (b.type === 'demolished') badgeClass = 'badge-b-dem';
            else if (b.type === 'rebuilt') badgeClass = 'badge-b-rebuilt';

            item.innerHTML = `
                <div class="building-card-header">
                    <span class="building-title">${b.icon} Bina #${b.id}</span>
                    <span class="badge-b-type ${badgeClass}">${b.type_tr}</span>
                </div>
                <div class="building-meta-row">
                    <span>Taban: <strong>${formatArea(b.area_m2, b.area_px)}</strong></span>
                    <span>Çevre: ${b.perimeter_m != null ? b.perimeter_m + ' m' : '—'}</span>
                    <span>Güven: %${b.confidence_pct}</span>
                </div>
            `;

            item.addEventListener('click', () => {
                focusOnBuilding(b);
            });

            el.buildingItemsList.appendChild(item);
        });
        if (existing.length > MAX_EXISTING_CARDS) {
            const more = document.createElement('div');
            more.className = 'building-card-more';
            more.textContent = `+${existing.length - MAX_EXISTING_CARDS} mevcut bina daha (görüntü üzerinde gösteriliyor)`;
            el.buildingItemsList.appendChild(more);
        }
    }

    // 5. Fit the stage to the image, reset the view & put the swipe divider in the middle
    fitStageToImage();
    resetStageView();
    updateSwipeDivider(50);
}

// Focus & Zoom to a specific building
function focusOnBuilding(b) {
    state.selectedBuildingId = b.id;

    // Highlight card in sidebar list
    document.querySelectorAll('.building-card-item').forEach(c => c.classList.remove('selected'));
    const card = document.getElementById(`card-building-${b.id}`);
    if (card) {
        card.classList.add('selected');
        card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // Highlight polygon in SVG
    document.querySelectorAll('.stage-vector-svg polygon').forEach(p => p.classList.remove('highlighted'));
    const poly = document.getElementById(`svg-poly-${b.id}`);
    if (poly) {
        poly.classList.add('highlighted');
        setTimeout(() => poly.classList.remove('highlighted'), 3000);
    }

    // Center Stage View on Building Centroid with Zoom
    if (b.centroid_px) {
        const imgSize = state.resultsData?.image_size || [512, 512];
        const cx = b.centroid_px[0];
        const cy = b.centroid_px[1];
        
        state.zoomScale = 2.0;
        // Offset (screen px) that brings (cx, cy) to the frame centre: image px -> stage px -> zoomed px
        const pxPerImagePx = (el.swipeStageContent?.offsetWidth || imgSize[0]) / imgSize[0];
        state.panX = (imgSize[0] / 2 - cx) * pxPerImagePx * state.zoomScale;
        state.panY = (imgSize[1] / 2 - cy) * pxPerImagePx * state.zoomScale;
        applyStageTransform();
    }

    // Display tooltip
    showBuildingTooltip(b);
    setTimeout(hideBuildingTooltip, 3000);
}

// Tooltip helpers
function showBuildingTooltip(b, event) {
    if (!el.buildingHoverTooltip) return;
    el.buildingHoverTooltip.innerHTML = `
        <div style="font-weight:700; margin-bottom:2px;">${b.icon} Bina #${b.id} - ${b.type_tr}</div>
        <div style="font-size:0.7rem; color:#cbd5e1; display:flex; gap:8px;">
            <span>Taban: <strong>${formatArea(b.area_m2, b.area_px)}</strong></span>
            <span>Çevre: ${b.perimeter_m != null ? b.perimeter_m + ' m' : '—'}</span>
            <span>Güven: %${b.confidence_pct}</span>
        </div>
    `;
    
    if (event && el.swipeStageContent) {
        const rect = el.swipeStageContent.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        el.buildingHoverTooltip.style.left = `${x}px`;
        el.buildingHoverTooltip.style.top = `${y}px`;
    } else if (b.centroid_px && el.swipeStageContent) {
        const imgSize = state.resultsData?.image_size || [512, 512];
        const pctX = (b.centroid_px[0] / imgSize[0]) * 100;
        const pctY = (b.centroid_px[1] / imgSize[1]) * 100;
        el.buildingHoverTooltip.style.left = `${pctX}%`;
        el.buildingHoverTooltip.style.top = `${pctY}%`;
    }
    
    el.buildingHoverTooltip.style.display = 'block';
}

function hideBuildingTooltip() {
    if (el.buildingHoverTooltip) {
        el.buildingHoverTooltip.style.display = 'none';
    }
}

// Swipe: the divider sits in the (unzoomed) stage frame at state.swipePosPct of its width, so it keeps
// its size and screen position while the image is zoomed or panned; the T1 clip follows it.
// Left shows T1 (1. İlk), right shows T2 (2. Sonraki).
function updateSwipeDivider(pct) {
    pct = Math.max(0, Math.min(100, pct));
    state.swipePosPct = pct;
    if (el.swipeDividerLine) {
        el.swipeDividerLine.style.left = `${pct}%`;
    }
    syncSwipeClip();
}

// Clip T1 at the divider's position expressed in the zoomed image's own coordinates
function syncSwipeClip() {
    if (!el.stageT1Clipper || !el.swipeStageOuter || !el.swipeStageContent) return;
    const frame = el.swipeStageOuter.getBoundingClientRect();
    const image = el.swipeStageContent.getBoundingClientRect();
    if (!image.width) return;
    const dividerX = frame.left + frame.width * state.swipePosPct / 100;
    const pct = Math.max(0, Math.min(100, (dividerX - image.left) / image.width * 100));
    el.stageT1Clipper.style.clipPath = `polygon(0 0, ${pct}% 0, ${pct}% 100%, 0 100%)`;
}

// Size the stage to the image's aspect ratio (non-square uploads were stretched into a square)
function fitStageToImage() {
    if (!el.swipeStageContent || !el.swipeStageOuter) return;
    const [w, h] = state.resultsData?.image_size || [512, 512];
    const frame = el.swipeStageOuter.getBoundingClientRect();
    if (!frame.width || !frame.height) return;
    const scale = Math.min(frame.width * 0.9 / w, frame.height * 0.9 / h);
    el.swipeStageContent.style.width = `${Math.round(w * scale)}px`;
    el.swipeStageContent.style.height = `${Math.round(h * scale)}px`;
    el.swipeStageContent.style.maxWidth = 'none';
    el.swipeStageContent.style.maxHeight = 'none';
    syncSwipeClip();
}

// Stage Zoom & Pan Transforms
function applyStageTransform() {
    if (!el.swipeStageContent) return;
    el.swipeStageContent.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.zoomScale})`;
    if (el.lblZoomLevel) {
        el.lblZoomLevel.textContent = `${Math.round(state.zoomScale * 100)}%`;
    }
    syncSwipeClip();
}

function setZoom(scale) {
    state.zoomScale = Math.max(0.8, Math.min(5.0, scale));
    applyStageTransform();
}

function resetStageView() {
    state.zoomScale = 1.0;
    state.panX = 0;
    state.panY = 0;
    applyStageTransform();
}

// Swipe Dragging & Stage Pan/Zoom Interactions
function setupSwipeInteractions() {
    // 1. Swipe Divider Dragging
    const handle = el.swipeDividerHandle;
    const divider = el.swipeDividerLine;

    const onSwipeMove = (clientX) => {
        if (!el.swipeStageOuter) return;
        const rect = el.swipeStageOuter.getBoundingClientRect();
        const offsetX = clientX - rect.left;
        let pct = (offsetX / rect.width) * 100;
        updateSwipeDivider(pct);
    };

    if (handle) {
        handle.addEventListener('mousedown', (e) => {
            e.preventDefault();
            e.stopPropagation();
            state.isDraggingSwipe = true;
            document.body.style.cursor = 'ew-resize';
        });
    }

    if (divider) {
        divider.addEventListener('mousedown', (e) => {
            e.preventDefault();
            e.stopPropagation();
            state.isDraggingSwipe = true;
            document.body.style.cursor = 'ew-resize';
        });
    }

    // Touch support for swipe divider
    if (handle) {
        handle.addEventListener('touchstart', (e) => {
            e.stopPropagation();
            state.isDraggingSwipe = true;
        }, { passive: true });
    }

    // 2. Stage Pan Interactions (Drag to Pan when zoomed or inspecting)
    if (el.swipeStageViewport) {
        el.swipeStageViewport.addEventListener('mousedown', (e) => {
            if (state.isDraggingSwipe) return;
            // Left mouse button pan
            if (e.button === 0) {
                state.isPanning = true;
                state.panStartX = e.clientX - state.panX;
                state.panStartY = e.clientY - state.panY;
                if (el.swipeStageViewport) el.swipeStageViewport.classList.add('grabbing');
            }
        });

        // Wheel Zoom (on the whole frame, so it also works over the divider)
        (el.swipeStageOuter || el.swipeStageViewport).addEventListener('wheel', (e) => {
            e.preventDefault();
            const delta = e.deltaY < 0 ? 0.15 : -0.15;
            setZoom(state.zoomScale + delta);
        }, { passive: false });
    }

    // Window Mouse Move & Up for Smooth Dragging
    window.addEventListener('mousemove', (e) => {
        if (state.isDraggingSwipe) {
            onSwipeMove(e.clientX);
        } else if (state.isPanning) {
            state.panX = e.clientX - state.panStartX;
            state.panY = e.clientY - state.panStartY;
            applyStageTransform();
        }
    });

    window.addEventListener('resize', fitStageToImage);

    window.addEventListener('mouseup', () => {
        if (state.isDraggingSwipe) {
            state.isDraggingSwipe = false;
            document.body.style.cursor = 'default';
        }
        if (state.isPanning) {
            state.isPanning = false;
            if (el.swipeStageViewport) el.swipeStageViewport.classList.remove('grabbing');
        }
    });

    window.addEventListener('touchmove', (e) => {
        if (state.isDraggingSwipe && e.touches[0]) {
            onSwipeMove(e.touches[0].clientX);
        }
    }, { passive: true });

    window.addEventListener('touchend', () => {
        state.isDraggingSwipe = false;
    });

    // 3. Zoom Toolbar Buttons
    if (el.btnZoomIn) {
        el.btnZoomIn.addEventListener('click', () => setZoom(state.zoomScale + 0.25));
    }
    if (el.btnZoomOut) {
        el.btnZoomOut.addEventListener('click', () => setZoom(state.zoomScale - 0.25));
    }
    if (el.btnZoomReset) {
        el.btnZoomReset.addEventListener('click', () => resetStageView());
    }

    // 4. Vector Detection Layer Toggles (User Request: "vector detection katmanları açılı kapanır olsun")
    if (el.chkLayerNew) {
        el.chkLayerNew.addEventListener('change', (e) => {
            if (el.svgGroupNew) el.svgGroupNew.style.display = e.target.checked ? '' : 'none';
        });
    }
    if (el.chkLayerRebuilt) {
        el.chkLayerRebuilt.addEventListener('change', (e) => {
            if (el.svgGroupRebuilt) el.svgGroupRebuilt.style.display = e.target.checked ? '' : 'none';
        });
    }
    if (el.chkLayerDem) {
        el.chkLayerDem.addEventListener('change', (e) => {
            if (el.svgGroupDem) el.svgGroupDem.style.display = e.target.checked ? '' : 'none';
        });
    }
    if (el.chkLayerExist) {
        el.chkLayerExist.addEventListener('change', (e) => {
            if (el.svgGroupExist) el.svgGroupExist.style.display = e.target.checked ? '' : 'none';
        });
    }
    if (el.chkLayerLabels) {
        el.chkLayerLabels.addEventListener('change', (e) => {
            if (el.svgGroupLabels) el.svgGroupLabels.style.display = e.target.checked ? '' : 'none';
        });
    }
}

// ==========================================
// MODALS & SETTINGS HELPERS
// ==========================================
function openModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) modal.classList.add('show');
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) modal.classList.remove('show');
}

function escapeHtml(str) {
    return (str || '').replace(/[&<>"']/g, function(m) {
        return ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[m];
    });
}

// Only the minimum building area is user-tunable: the ML engine's probability threshold had no
// measurable effect (F1 0.726 / 0.725 / 0.722 at 0.4 / 0.5 / 0.6), so that setting was removed.
function getAiSettings() {
    try {
        const saved = localStorage.getItem('atlas_ai_settings');
        if (saved) return JSON.parse(saved);
    } catch(e) {}
    return { minArea: 30.0 };
}

function initSettings() {
    const s = getAiSettings();
    if (el.rangeSettingMinarea) {
        el.rangeSettingMinarea.value = s.minArea;
        if (el.lblSettingMinarea) el.lblSettingMinarea.textContent = `${s.minArea} m²`;
        el.rangeSettingMinarea.addEventListener('input', (e) => {
            if (el.lblSettingMinarea) el.lblSettingMinarea.textContent = `${e.target.value} m²`;
        });
    }
    if (el.btnSaveSettings) {
        el.btnSaveSettings.addEventListener('click', () => {
            const minArea = parseFloat(el.rangeSettingMinarea.value);
            localStorage.setItem('atlas_ai_settings', JSON.stringify({ minArea }));
            closeModal('modal-settings');
            showToast('Model ve tespit ayarları kaydedildi!', 'success');
        });
    }
    if (el.btnClearAllProjects) {
        el.btnClearAllProjects.addEventListener('click', async () => {
            const confirmed = await showConfirm({
                title: 'Veritabanını Sıfırla',
                message: 'Tüm kayıtlı projeleri silmek ve veritabanını sıfırlamak istediğinize emin misiniz? Bu işlem geri alınamaz.',
                icon: 'danger',
                confirmText: 'Evet, Sıfırla',
                cancelText: 'Vazgeç'
            });
            if (!confirmed) return;

            try {
                const res = await fetch('/api/projects/clear', { method: 'POST' });
                const data = await res.json();
                if (data.success) {
                    closeModal('modal-settings');
                    loadProjects();
                    showToast('Tüm projeler başarıyla sıfırlandı.', 'success');
                }
            } catch(e) {
                showToast('Projeler sıfırlanırken hata oluştu.', 'error');
            }
        });
    }
}

function initModals() {
    if (el.menuBtnDiscover) {
        el.menuBtnDiscover.addEventListener('click', (e) => {
            e.preventDefault();
            openModal('modal-discover');
        });
    }
    if (el.menuBtnSettings) {
        el.menuBtnSettings.addEventListener('click', (e) => {
            e.preventDefault();
            openModal('modal-settings');
        });
    }

    // Discover quick analyze buttons
    document.querySelectorAll('.btn-quick-analyze').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const card = btn.closest('.discover-card');
            const hotspotId = card.getAttribute('data-hotspot');
            closeModal('modal-discover');
            
            createNewProject('live-hotspot');
            if (el.selectHotspot) {
                el.selectHotspot.value = hotspotId;
                updateHotspotCard(hotspotId);
            }
            handleStep1Next();
        });
    });

    // Global modal close handlers
    document.addEventListener('click', (e) => {
        const closeBtn = e.target.closest('[data-close]');
        if (closeBtn) {
            const modalId = closeBtn.getAttribute('data-close');
            closeModal(modalId);
            return;
        }
        if (e.target.classList.contains('atlas-modal-backdrop')) {
            e.target.classList.remove('show');
        }
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            document.querySelectorAll('.atlas-modal-backdrop.show').forEach(m => m.classList.remove('show'));
        }
    });
}

// ==========================================
// SETUP EVENT LISTENERS
// ==========================================
function setupEvents() {
    // 1. Dashboard Buttons
    if (el.btnSidebarCreateProject) {
        el.btnSidebarCreateProject.addEventListener('click', () => createNewProject());
    }
    if (el.btnBannerNewProject) {
        el.btnBannerNewProject.addEventListener('click', () => createNewProject());
    }
    if (el.btnShortcutCreate) {
        el.btnShortcutCreate.addEventListener('click', () => createNewProject());
    }
    if (el.btnShortcutImport) {
        el.btnShortcutImport.addEventListener('click', () => createNewProject('custom-upload'));
    }
    if (el.btnShortcutTemplates) {
        el.btnShortcutTemplates.addEventListener('click', () => createNewProject('live-hotspot'));
    }

    // Projects menu item
    if (el.menuBtnProjects) {
        el.menuBtnProjects.addEventListener('click', (e) => {
            e.preventDefault();
            switchScreen('dashboard');
        });
    }

    // Search Projects in Dashboard
    if (el.inputDashSearch) {
        let debounceTimer;
        el.inputDashSearch.addEventListener('input', (e) => {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => {
                loadProjects(e.target.value);
            }, 250);
        });
    }

    // Filter Scope in Dashboard
    if (el.dashFilterScope) {
        el.dashFilterScope.addEventListener('change', (e) => {
            state.activeFilterScope = e.target.value;
            applyFilterAndSort();
        });
    }

    // Sort By in Dashboard
    if (el.dashSortBy) {
        el.dashSortBy.addEventListener('change', (e) => {
            state.activeSortBy = e.target.value;
            applyFilterAndSort();
        });
    }

    // View Mode Toggle (Grid vs List)
    if (el.btnViewGrid) {
        el.btnViewGrid.addEventListener('click', () => setViewMode('grid'));
    }
    if (el.btnViewList) {
        el.btnViewList.addEventListener('click', () => setViewMode('list'));
    }

    // Back to Dashboard Button
    if (el.btnBackToDashboard) {
        el.btnBackToDashboard.addEventListener('click', () => {
            switchScreen('dashboard');
        });
    }

    // Project Name Sync
    if (el.inputTopbarProjectName && el.inputStep1ProjectName) {
        el.inputTopbarProjectName.addEventListener('input', (e) => {
            el.inputStep1ProjectName.value = e.target.value;
            setSaveIndicator(false);
        });
        el.inputStep1ProjectName.addEventListener('input', (e) => {
            el.inputTopbarProjectName.value = e.target.value;
            setSaveIndicator(false);
        });
    }

    // Manual Save Button
    if (el.btnManualSaveProject) {
        el.btnManualSaveProject.addEventListener('click', () => {
            saveProjectToDatabase(true);
        });
    }

    // Toggle Building Polygons Visibility
    if (el.btnTogglePolygons) {
        el.btnTogglePolygons.addEventListener('click', () => {
            state.allVectorsVisible = !state.allVectorsVisible;
            if (el.stageVectorSvg) {
                el.stageVectorSvg.style.display = state.allVectorsVisible ? 'block' : 'none';
            }
            if (el.lblTogglePoly) {
                el.lblTogglePoly.textContent = state.allVectorsVisible ? 'Tüm Vektörleri Gizle' : 'Tüm Vektörleri Göster';
            }
            if (el.btnTogglePolygons) {
                el.btnTogglePolygons.classList.toggle('active', !state.allVectorsVisible);
            }
        });
    }

    // Source Card Selection
    el.sourceCards.forEach(card => {
        card.addEventListener('click', () => {
            el.sourceCards.forEach(c => c.classList.remove('active'));
            card.classList.add('active');
            
            const src = card.getAttribute('data-source');
            state.sourceType = src;
            
            Object.keys(el.panels).forEach(k => {
                if (el.panels[k]) {
                    el.panels[k].style.display = (k === src) ? 'block' : 'none';
                }
            });

            if (src === 'map-click') {
                initSelectMap();
                setTimeout(() => {
                    if (state.selectMap) {
                        state.selectMap.invalidateSize();
                    }
                }, 100);
            }
        });
    });

    // Hotspot Selection
    if (el.selectHotspot) {
        el.selectHotspot.addEventListener('change', (e) => {
            updateHotspotCard(e.target.value);
        });
    }

    // Step 1 -> Step 2
    if (el.btnGotoStep2) {
        el.btnGotoStep2.addEventListener('click', handleStep1Next);
    }

    // Step 2 -> Step 1
    if (el.btnBackToStep1) {
        el.btnBackToStep1.addEventListener('click', () => goToStep(1));
    }

    // Step 2 -> Step 3
    if (el.btnRunBuildingDetection) {
        el.btnRunBuildingDetection.addEventListener('click', handleRunDetection);
    }

    // Restart Wizard
    if (el.btnRestartWizard) {
        el.btnRestartWizard.addEventListener('click', () => {
            state.resultsData = null;
            goToStep(1);
        });
    }

    // Export Downloads
    if (el.btnExportGeoJson) {
        el.btnExportGeoJson.addEventListener('click', () => {
            window.location.href = '/api/export/geojson';
        });
    }
    if (el.btnExportCsv) {
        el.btnExportCsv.addEventListener('click', () => {
            window.location.href = '/api/export/csv';
        });
    }

    // Swipe Interactions
    setupSwipeInteractions();
}

// Bootstrap
document.addEventListener('DOMContentLoaded', () => {
    setupEvents();
    initModals();
    initSettings();
    setViewMode(state.viewMode);
    loadHotspots();
    loadProjects();
});
