const express = require("express");
const { MongoClient } = require("mongodb");

const app = express();
app.use(express.json());

const PORT = process.env.PORT || 3000;
const MONGO_URL =
  process.env.MONGO_URL ||
  "mongodb://admin:password@localhost:27017/?authSource=admin";

const client = new MongoClient(MONGO_URL);

async function start() {
  try {
    await client.connect();
    console.log("✅ Connected to MongoDB");

    app.get("/", (req, res) => {
      res.json({
        status: "ok",
        service: "profile-backend",
      });
    });

    app.listen(PORT, () => {
      console.log(`🚀 Backend running on port ${PORT}`);
    });
  } catch (err) {
    console.error("❌ Mongo connection failed", err);
    process.exit(1);
  }
}

start();
