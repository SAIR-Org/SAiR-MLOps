const express = require('express');
const { MongoClient } = require('mongodb');

const app = express();

/* =========================
   Middleware
========================= */
app.use(express.json());
app.use(express.static('.'));

/* =========================
   MongoDB config
========================= */
const MONGO_URL = 'mongodb://admin:password@localhost:27017/?authSource=admin';
const DB_NAME = 'profile_app';
const COLLECTION = 'profiles';

let collection;

/* =========================
   Connect to MongoDB
========================= */
async function connectDB() {
    const client = new MongoClient(MONGO_URL);
    await client.connect();
    console.log('✅ Connected to MongoDB');

    const db = client.db(DB_NAME);
    collection = db.collection(COLLECTION);
}

/* =========================
   Routes
========================= */

// Get profile
app.get('/profile', async (req, res) => {
    try {
        const profile = await collection.findOne({ _id: 'single' });
        res.json(profile || { name: '', email: '', language: '' });
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: 'Failed to load profile' });
    }
});

// Save profile
app.post('/profile', async (req, res) => {
    const { name, email, language } = req.body;

    const profile = {
        _id: 'single',
        name: name || '',
        email: email || '',
        language: language || ''
    };

    try {
        await collection.updateOne(
            { _id: 'single' },
            { $set: profile },
            { upsert: true }
        );

        console.log('✅ Profile saved:', profile);
        res.json({ message: 'Profile saved' });
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: 'Failed to save profile' });
    }
});

// Health check
app.get('/health', (req, res) => {
    res.json({
        status: 'running',
        db: 'mongodb',
        persistence: true
    });
});

/* =========================
   Start server
========================= */
const PORT = 3000;

connectDB()
    .then(() => {
        app.listen(PORT, () => {
            console.log('=======================================');
            console.log(`🚀 Server running at http://localhost:${PORT}`);
            console.log('✅ Storage: MongoDB (persistent)');
            console.log('=======================================');
        });
    })
    .catch(err => {
        console.error('❌ Failed to start server:', err);
        process.exit(1);
    });
