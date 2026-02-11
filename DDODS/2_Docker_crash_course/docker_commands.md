# Docker Commands

## 1. Clean start

```bash
docker rm -f $(docker ps -aq)
docker network rm profile-net
```

---

## 2. Create network

```bash
docker network create profile-net
```

---

## 3. Run MongoDB

```bash
docker run -d \
  --name mongodb \
  --net profile-net \
  -p 27017:27017 \
  -e MONGO_INITDB_ROOT_USERNAME=admin \
  -e MONGO_INITDB_ROOT_PASSWORD=password \
  mongo
```

---

## 4. Run mongo-express

```bash
docker run -d \
  --name mongo-express \
  --net profile-net \
  -p 8081:8081 \
  -e ME_CONFIG_MONGODB_URL="mongodb://admin:password@mongodb:27017/?authSource=admin" \
  mongo-express
```

---

## 5. Run backend (host)

```bash
npm install
node server.js
```

---

## 6. Access

* App: [http://localhost:3000](http://localhost:3000)
* API: [http://localhost:3000/profile](http://localhost:3000/profile)
* Mongo UI: [http://localhost:8081](http://localhost:8081)

  * user: `admin`
  * pass: `pass`

---

## 7. Data location

mongo-express →
`profile_app` → `profiles`


## Note : 
Please refer to the offical docs if you faced any issue with the above commands.
maybe there is an update in the image or something else. 