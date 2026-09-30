import os
import sqlite3
import time
import uuid
import datetime
import json

class ProjectManager:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.db_path = os.path.join(self.data_dir, 'projects.db')
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    description TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    created_at_formatted TEXT,
                    updated_at_formatted TEXT,
                    source_type TEXT,
                    location_name TEXT,
                    year_t1 TEXT,
                    year_t2 TEXT,
                    thumbnail_url TEXT,
                    is_shared INTEGER,
                    zoom INTEGER,
                    coords TEXT,
                    stats TEXT,
                    results_data TEXT
                )
            ''')
            conn.commit()

    def _row_to_dict(self, row):
        d = dict(row)
        # Parse JSON fields
        for field in ['coords', 'stats', 'results_data']:
            if d.get(field):
                try:
                    d[field] = json.loads(d[field])
                except Exception:
                    d[field] = None
        # Convert integer to boolean
        if 'is_shared' in d:
            d['is_shared'] = bool(d['is_shared'])
        return d

    def list_projects(self, search=None):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = '''
                SELECT id, name, description, created_at, updated_at, 
                       created_at_formatted, updated_at_formatted, source_type, 
                       location_name, year_t1, year_t2, thumbnail_url, 
                       is_shared, zoom, coords, stats 
                FROM projects 
                ORDER BY updated_at DESC, created_at DESC
            '''
            cursor.execute(query)
            rows = cursor.fetchall()
            
            result = []
            for row in rows:
                p = self._row_to_dict(row)
                if search:
                    s = search.lower()
                    if s not in (p.get('name') or '').lower() and s not in (p.get('description') or '').lower():
                        continue
                # For listings, results_data is omitted to keep it lightweight.
                # However, the rest is maintained.
                result.append(p)
                
            return result

    def get_project(self, project_id):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM projects WHERE id = ?', (project_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_dict(row)
            return None

    def save_project(self, project_data):
        now = datetime.datetime.now()
        now_iso = now.isoformat()
        now_formatted = now.strftime('%d.%m.%Y, %H:%M')

        project_id = project_data.get('id')
        if not project_id:
            project_id = f"proj_{int(time.time())}_{uuid.uuid4().hex[:6]}"
            project_data['id'] = project_id
            project_data['created_at'] = now_iso
            project_data['created_at_formatted'] = now_formatted

        project_data['updated_at'] = now_iso
        project_data['updated_at_formatted'] = now_formatted

        if not project_data.get('thumbnail_url') and project_data.get('results_data'):
            overlays = project_data.get('results_data', {}).get('overlays', {})
            project_data['thumbnail_url'] = overlays.get('t2_png_base64') or overlays.get('t1_png_base64') or ''

        # Extract values
        values = (
            project_data['id'],
            project_data.get('name', 'İsimsiz Proje'),
            project_data.get('description', ''),
            project_data.get('created_at'),
            project_data.get('updated_at'),
            project_data.get('created_at_formatted', ''),
            project_data.get('updated_at_formatted', ''),
            project_data.get('source_type', 'live-hotspot'),
            project_data.get('location_name', ''),
            project_data.get('year_t1', ''),
            project_data.get('year_t2', ''),
            project_data.get('thumbnail_url', ''),
            1 if project_data.get('is_shared') else 0,
            project_data.get('zoom', 17),
            json.dumps(project_data.get('coords', [0, 0])),
            json.dumps(project_data.get('stats', {})),
            json.dumps(project_data.get('results_data', {}))
        )

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO projects (
                    id, name, description, created_at, updated_at, 
                    created_at_formatted, updated_at_formatted, source_type, 
                    location_name, year_t1, year_t2, thumbnail_url, 
                    is_shared, zoom, coords, stats, results_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    description=excluded.description,
                    updated_at=excluded.updated_at,
                    updated_at_formatted=excluded.updated_at_formatted,
                    source_type=excluded.source_type,
                    location_name=excluded.location_name,
                    year_t1=excluded.year_t1,
                    year_t2=excluded.year_t2,
                    thumbnail_url=excluded.thumbnail_url,
                    is_shared=excluded.is_shared,
                    zoom=excluded.zoom,
                    coords=excluded.coords,
                    stats=excluded.stats,
                    results_data=excluded.results_data
            ''', values)
            conn.commit()

        return project_data

    def delete_project(self, project_id):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM projects WHERE id = ?', (project_id,))
            changes = cursor.rowcount
            conn.commit()
            return changes > 0

    def clear_all(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM projects')
            conn.commit()
            return True
