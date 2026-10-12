const express = require('express');
const sqlite3 = require('sqlite3').verbose();
const path = require('path');
const cors = require('cors');
const fs = require('fs');

const app = express();
app.use(express.json());
app.use(cors());

// Initialize SQLite database
const dbPath = path.join(__dirname, 'tracksense.db');
const db = new sqlite3.Database(dbPath, (err) => {
  if (err) console.error('Database connection error:', err);
  else console.log('SQLite database connected');
});

// Create tables if they don't exist
db.serialize(() => {
  db.run(`
    CREATE TABLE IF NOT EXISTS inspections (
      id TEXT PRIMARY KEY,
      mode TEXT,
      section TEXT,
      startKP TEXT,
      endKP TEXT,
      distance REAL,
      date TEXT,
      status TEXT,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
  `);

  db.run(`
    CREATE TABLE IF NOT EXISTS detections (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      inspection_id TEXT,
      type TEXT,
      severity TEXT,
      chainage TEXT,
      confidence INTEGER,
      FOREIGN KEY (inspection_id) REFERENCES inspections(id)
    )
  `);

  db.run(`
    CREATE TABLE IF NOT EXISTS defects (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      inspection_id TEXT,
      defect_type TEXT,
      count INTEGER,
      FOREIGN KEY (inspection_id) REFERENCES inspections(id)
    )
  `);
});

// Helper: promisify db.run
const dbRun = (sql, params = []) => {
  return new Promise((resolve, reject) => {
    db.run(sql, params, function(err) {
      if (err) reject(err);
      else resolve(this);
    });
  });
};

// Helper: promisify db.get
const dbGet = (sql, params = []) => {
  return new Promise((resolve, reject) => {
    db.get(sql, params, (err, row) => {
      if (err) reject(err);
      else resolve(row);
    });
  });
};

// Helper: promisify db.all
const dbAll = (sql, params = []) => {
  return new Promise((resolve, reject) => {
    db.all(sql, params, (err, rows) => {
      if (err) reject(err);
      else resolve(rows || []);
    });
  });
};

// ===== ROUTES =====

// Get dashboard stats
app.get('/api/stats', async (req, res) => {
  try {
    const totalInspections = await dbGet('SELECT COUNT(*) as count FROM inspections');
    const totalDistance = await dbGet('SELECT COALESCE(SUM(distance), 0) as total FROM inspections');
    const recentInspections = await dbAll('SELECT * FROM inspections ORDER BY created_at DESC LIMIT 5');
    
    res.json({
      totalInspections: totalInspections.count,
      totalDistance: totalDistance.total,
      recentCount: recentInspections.length,
      recent: recentInspections
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Get all inspections
app.get('/api/inspections', async (req, res) => {
  try {
    const inspections = await dbAll('SELECT * FROM inspections ORDER BY created_at DESC');
    res.json(inspections);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Get single inspection with detections and defects
app.get('/api/inspections/:id', async (req, res) => {
  try {
    const inspection = await dbGet('SELECT * FROM inspections WHERE id = ?', [req.params.id]);
    if (!inspection) {
      return res.status(404).json({ error: 'Inspection not found' });
    }

    const detections = await dbAll('SELECT * FROM detections WHERE inspection_id = ?', [req.params.id]);
    const defects = await dbAll('SELECT defect_type, count FROM defects WHERE inspection_id = ?', [req.params.id]);

    inspection.detections = detections;
    inspection.defectsMap = {};
    defects.forEach(d => {
      inspection.defectsMap[d.defect_type] = d.count;
    });

    res.json(inspection);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Create new inspection
app.post('/api/inspections', async (req, res) => {
  try {
    const { id, mode, section, startKP, targetDistance } = req.body;
    
    await dbRun(
      'INSERT INTO inspections (id, mode, section, startKP, distance, date, status) VALUES (?, ?, ?, ?, ?, ?, ?)',
      [id, mode, section, startKP, targetDistance, new Date().toISOString().split('T')[0], 'active']
    );

    res.json({ success: true, id });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Add detection to inspection
app.post('/api/inspections/:id/detections', async (req, res) => {
  try {
    const { type, severity, chainage, confidence } = req.body;
    
    await dbRun(
      'INSERT INTO detections (inspection_id, type, severity, chainage, confidence) VALUES (?, ?, ?, ?, ?)',
      [req.params.id, type, severity, chainage, confidence]
    );

    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Complete inspection
app.post('/api/inspections/:id/complete', async (req, res) => {
  try {
    const { endKP, defects } = req.body;
    
    // Update inspection
    await dbRun(
      'UPDATE inspections SET endKP = ?, status = ? WHERE id = ?',
      [endKP, 'completed', req.params.id]
    );

    // Save defects
    for (const [defectType, count] of Object.entries(defects)) {
      await dbRun(
        'INSERT INTO defects (inspection_id, defect_type, count) VALUES (?, ?, ?)',
        [req.params.id, defectType, count]
      );
    }

    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Serve static files (frontend)
app.use(express.static(path.join(__dirname, 'public')));

// Serve the main HTML file
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok' });
});

// Start server
const PORT = process.env.PORT || 3000;
app.post('/predict', express.raw({ type: '*/*', limit: '20mb' }), async (req, res) => {
  try {
    const r = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: { 'content-type': req.headers['content-type'] },
      body: req.body,
    });
    res.status(r.status).type('json').send(await r.text());
  } catch (e) {
    res.status(502).json({ error: String(e) });
  }
});
app.listen(PORT, () => {
  console.log(`TrackSense backend running on port ${PORT}`);
});

// Graceful shutdown
process.on('SIGINT', () => {
  db.close((err) => {
    if (err) console.error(err);
    process.exit(0);
  });
});
