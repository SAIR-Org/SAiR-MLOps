const express = require('express');
const { MongoClient } = require('mongodb');

const app = express();
app.use(express.json());
app.use(express.static('.'));

const MONGO_URL = process.env.MONGO_URL || 'mongodb://localhost:27017';
const DB_NAME = 'profile_app';
const COLLECTION = 'profiles';

let collection = null;

async function connectDB() {
    console.log(`Attempting to connect to: ${MONGO_URL}`);
    try {
        const client = new MongoClient(MONGO_URL, {
            serverSelectionTimeoutMS: 5000,
            connectTimeoutMS: 10000
        });
        await client.connect();
        console.log('✅ Connected to MongoDB');
        const db = client.db(DB_NAME);
        collection = db.collection(COLLECTION);
        
        // Create collection if it doesn't exist
        const cols = await db.listCollections({ name: COLLECTION }).toArray();
        if (cols.length === 0) {
            await db.createCollection(COLLECTION);
            console.log(`Created collection: ${COLLECTION}`);
        }
    } catch (err) {
        console.log('❌ MongoDB connection failed:', err.message);
        console.log('⚠️  Running in fallback mode (no persistence)');
        
        // Fallback in-memory storage
        let memoryStore = { name: '', email: '', language: '' };
        collection = {
            findOne: async () => memoryStore,
            updateOne: async (query, update) => {
                memoryStore = { ...memoryStore, ...update.$set };
                console.log('📝 Saved to memory:', memoryStore);
                return { modifiedCount: 1 };
            }
        };
    }
}

// Routes
app.get('/profile', async (req, res) => {
    const profile = await collection.findOne({ _id: 'single' });
    res.json(profile || { name: '', email: '', language: '' });
});

app.post('/profile', async (req, res) => {
    const profile = {
        _id: 'single',
        name: req.body.name || '',
        email: req.body.email || '',
        language: req.body.language || ''
    };
    
    await collection.updateOne(
        { _id: 'single' },
        { $set: profile },
        { upsert: true }
    );
    
    res.json({ message: 'Profile saved', storage: collection.db ? 'MongoDB' : 'Memory' });
});

app.get('/health', (req, res) => {
    res.json({
        status: 'running',
        db: collection.db ? 'connected' : 'memory',
        timestamp: new Date().toISOString()
    });
});

// Initialize and start
const PORT = 3000;
connectDB().then(() => {
    app.listen(PORT, () => {
        console.log(`🚀 Server running on port ${PORT}`);
        console.log(`📊 Health check: http://localhost:${PORT}/health`);
        console.log(`🔗 MongoDB URL: ${MONGO_URL}`);
    });
});