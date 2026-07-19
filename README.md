# ⚙️ SAiR MLOps Blueprint

<div align="center">

![SAiR MLOps Blueprint Banner](banner.jpg)

### **90% of ML Models Never Reach Production. This Is How You Become Someone Who Ships the Other 10%.**
*Module 6 of the SAIR Jr. Certification Track — Sudanese Artificial Intelligence Research (SAIR) Initiative*

<table>
<tr>
<td align="center">
<a href="https://t.me/sair19969">
<img src="https://img.shields.io/badge/JOIN_THE_LIVE_COHORT-0088CC?style=for-the-badge&logo=telegram&logoColor=white" alt="Telegram Community"/>
</a>
</td>
<td align="center">
<a href="https://youtube.com/playlist?list=PLVM9Nqm8zLE0&si=jtIah3TJB8PjOMgu">
<img src="https://img.shields.io/badge/WATCH_THE_THEORY-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="MLOps from First Principles"/>
</a>
</td>
<td align="center">
<a href="https://github.com/SAIR-Org/SAIR_Jr">
<img src="https://img.shields.io/badge/PART_OF-SAIR_Jr-9C27B0?style=for-the-badge&logo=github&logoColor=white" alt="SAIR Jr"/>
</a>
</td>
</tr>
</table>

**Duration:** 6-8 weeks · **Prerequisite:** [SAIR Jr. Modules 0-5](https://github.com/SAIR-Org/SAIR_Jr) · **Format:** Theory playlist + live cohort builds + required reading (+ optional proof-of-skill challenge)

</div>

---

## 📚 Table of Contents

- [🎯 The Problem With How MLOps Is Usually Taught](#-the-problem-with-how-mlops-is-usually-taught)
- [🗺️ Quick Navigation](#️-quick-navigation)
- [⚡ Why Two Tracks, Not One](#-why-two-tracks-not-one)
- [📅 Optional Track — Prove It: 100 Days of MLOps](#-optional-track--prove-it-100-days-of-mlops)
- [🏗️ SAIRCAMP Is a Series, Not a Single System](#️-saircamp-is-a-series-not-a-single-system)
- [🚕 A Closer Look — Project 1: SAIRCAMP MLOps](#-a-closer-look--project-1-saircamp-mlops)
- [🛠️ Technology Stack You'll Master](#️-technology-stack-youll-master)
- [💼 What You Walk Away Able to Do](#-what-you-walk-away-able-to-do)
- [📖 Required Reading](#-required-reading)
- [🎓 Module 6 Completion Requirements](#-module-6-completion-requirements)
- [🚀 Getting Started](#-getting-started)
- [📚 Where This Fits in SAIR Jr.](#-where-this-fits-in-sair-jr)

---

## 🎯 **The Problem With How MLOps Is Usually Taught**

You already know how to train a model. That was never the hard part.

The hard part is everything nobody puts in a tutorial: the model that works perfectly on your laptop and silently degrades in production. The experiment you can't reproduce three weeks later. The "quick deploy" that becomes a 2am pager alert. The world shifts — a pandemic, a policy change, a new customer segment — and your model keeps confidently returning answers that are now wrong, and nothing tells you.

Most courses respond to this by teaching you **tool syntax in isolation**: here's a 20-minute MLflow demo, here's a Docker "hello world." You walk away able to name the tools. You still can't explain what breaks without them, and you've never felt what it's like when the ground shifts under a model you shipped.

**This module is built backwards from that failure.** You don't learn MLflow because it's Tuesday. You learn it because Module 1 leaves you with 20 untracked experiments and no way to know which one was actually best — and MLflow is the only way out of the hole you're already in.

---

## 🗺️ **Quick Navigation**

| Track | Format | Where | Status |
|---|---|---|---|
| 📖 MLOps from First Principles | Solo-recorded playlist, concept-first | [YouTube](https://youtube.com/playlist?list=PLVM9Nqm8zLE0&si=jtIah3TJB8PjOMgu) · [Repo](https://github.com/SAIR-Org/MLOps-from-First-Principles) | ✅ Available |
| 🏗️ SAIRCAMP — Project 1: Classical ML | Live cohort, released week by week | Repo 🔒 not public yet | ✅ Built |
| 🧠 SAIRCAMP — Project 2: Deep Learning | Live cohort, revealed lecture by lecture | Repo 🔜 TBA | ✅ Built |
| 📄 SAIRCAMP — Project 3: Document Intelligence | Live cohort, scope TBA | Repo 🔜 Coming | 📝 Planned |
| 📅 *(Optional)* 100 Days of MLOps | Self-paced daily tasks, auto-validated | [KodeKloud](https://kodekloud.com/100-days-of-mlops) | ✅ Available (external, free) |

> This repo (`SAiR-MLOps-Blueprint`) is the hub — it maps the whole module and links out to every resource. It is **not** where the code lives; each SAIRCAMP project ships in its own repo, linked here as it's released.

---

## ⚡ **Why Two Tracks, Not One**

<div align="center">

### **🔥 Concepts in Isolation Don't Survive Contact With Production**

<table>
<tr>
<td width="100%" align="center">
<h3>"نظرية بلا تطبيق عرجاء، وتطبيق بلا نظرية أعمى"</h3>
<p><em>Theory without practice limps. Practice without theory is blind.</em></p>
</td>
</tr>
</table>

</div>

**MLOps is too wide a subject for one format to teach well**, so this module runs on two resources at once — not as alternatives, but as two lenses on the same subject, mirroring the exact split that makes the whole SAIR Jr. track work:

```mermaid
flowchart LR
    Start(["You"]) --> FP["MLOps from First Principles<br/>Solo YouTube playlist<br/>Concept-first, broad coverage<br/>Answers: why does this exist?"]
    Start --> SC["SAIRCAMP<br/>Live cohort sessions<br/>Project-first, production depth<br/>Answers: how do I build it?"]

    FP --> Merge["Watch the lecture,<br/>then build it live"]
    SC --> Merge

    Merge --> Outcome["A model that actually<br/>survives production"]

    style FP fill:#e3f2fd,stroke:#1565c0,color:#000
    style SC fill:#fff3e0,stroke:#e65100,color:#000
    style Merge fill:#238636,color:#fff
    style Outcome fill:#8957e5,color:#fff
```

**Recommended order:** watch the matching lecture in *First Principles* before the live build session in *SAIRCAMP* — the video gives you the mental model before you're in the weeds wiring Docker networks together.

---

## 📅 **Optional Track — Prove It: 100 Days of MLOps**

The two tracks above are what teach you MLOps. This one doesn't teach you anything new — it's where you go to **prove, publicly, that it stuck.**

**[100 Days of MLOps](https://kodekloud.com/100-days-of-mlops)** (KodeKloud) is a free challenge of 100 short, auto-validated, hands-on tasks across the MLOps stack — data versioning, experiment tracking, orchestration, model serving, monitoring, GPU training, canary deploys, GitOps, and more.

```mermaid
flowchart LR
    FP["First Principles<br/>why does this exist?"] --> SC["SAIRCAMP<br/>how do I build the whole thing?"]
    SC --> HD["100 Days of MLOps<br/>optional, but recommended:<br/>public proof of skill"]

    style FP fill:#e3f2fd,stroke:#1565c0,color:#000
    style SC fill:#fff3e0,stroke:#e65100,color:#000
    style HD fill:#e8f5e9,stroke:#2e7d32,color:#000
```

**Where it fits:** it comes *after* you already know the material from SAIRCAMP and First Principles — think of it less as a learning track and more as a public record you hand to a hiring manager. SAIRCAMP gives you one deep, real system you built once. 100 Days of MLOps gives you 100 completed, independently-validated tasks and a KodeKloud verified badge sitting next to it — evidence you can point to that isn't just "trust me, I built this."

**Why it's optional but highly recommended:**
- **It's proof, not just practice** — every task is auto-validated, so "I completed 100 Days of MLOps" isn't a self-graded claim. It's the closest thing to a receipt for your skills that a recruiter or hiring manager can actually check.
- **Zero setup cost** — every environment is provisioned for you, no local installs, no VPS bill. Nothing stopping you from starting the day you finish SAIRCAMP.
- **Broadens what you can point to in an interview** — it touches tools SAIRCAMP's NYC Taxi build doesn't go deep on (e.g. DVC, Feast, Argo Workflows, Kubernetes-based canary deploys), so it lets you say "I've also worked with X" beyond the one project.
- **Free** — no cost, no reason not to do it once you've completed the required tracks.

It's optional because SAIRCAMP + First Principles + the required reading are what the module actually certifies you on. But if you're serious about the job search, treat this as strongly recommended, not an afterthought.

---

## 🏗️ **SAIRCAMP Is a Series, Not a Single System**

This is the part that trips people up: **SAIRCAMP isn't one project.** It's a track of complete, real, end-to-end builds — each one a full production system on its own, each teaching MLOps against a different kind of model, released live to the cohort one at a time.

```mermaid
flowchart TD
    Hub["SAIRCAMP Series"] --> P1["Project 1 - SAIRCAMP MLOps<br/>NYC Taxi trip duration<br/>Classical ML, tabular data<br/>8 modules, notebook to production"]
    Hub --> P2["Project 2 - SAIRCAMP DL<br/>Deep learning system<br/>Scope TBA"]
    Hub --> P3["Project 3 - SAIRCAMP DocIntel<br/>Document Intelligence platform<br/>Scope TBA"]

    style P1 fill:#238636,color:#fff
    style P2 fill:#238636,color:#fff
    style P3 fill:#6e7681,color:#fff
```

<div align="center">

<table>
<thead>
<tr>
<th width="8%">#</th>
<th width="27%">Project</th>
<th width="35%">What It's Built Around</th>
<th width="15%">Status</th>
<th width="15%">Repo</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center"><strong>1</strong><br/>🚕</td>
<td><strong>SAIRCAMP MLOps</strong></td>
<td>NYC Taxi trip duration prediction — classical ML, tabular data, 8 modules from notebook to secured production system</td>
<td align="center">✅ Built<br/><sub>released week by week</sub></td>
<td align="center">🔒 not public yet</td>
</tr>
<tr>
<td align="center"><strong>2</strong><br/>🧠</td>
<td><strong>SAIRCAMP DL</strong></td>
<td>Deep learning system — full scope TBA</td>
<td align="center">✅ Built<br/><sub>revealed lecture by lecture</sub></td>
<td align="center">🔜 to be announced</td>
</tr>
<tr>
<td align="center"><strong>3</strong><br/>📄</td>
<td><strong>SAIRCAMP DocIntel</strong></td>
<td>Document Intelligence platform for digital transformation — scope TBA</td>
<td align="center">📝 Planned</td>
<td align="center">🔜 coming</td>
</tr>
</tbody>
</table>

</div>

> Repos go live as each project is revealed to the cohort. This table updates with real links as that happens — follow [Telegram](https://t.me/sair19969) for announcements.

Each project is a **complete arc on its own** — same MLOps principles, applied to a different problem shape, so the concepts generalize instead of feeling tied to one dataset.

---

## 🚕 **A Closer Look — Project 1: SAIRCAMP MLOps**

```mermaid
flowchart LR
    A["Notebook"] --> B["Tracked"]
    B --> C["Structured"]
    C --> D["Orchestrated"]
    D --> E["Served<br/>Online"]
    E --> F["Served Offline<br/>+ Monitored"]
    F --> G["Integrated<br/>System"]
    G --> H["Real<br/>VPS"]
    H --> I["Secured +<br/>Auto-Deploying"]

    classDef stage fill:#238636,color:#fff,stroke:#0f4b1e
    class A,B,C,D,E,F,G,H,I stage
```

| Module | What's Built | Key Concepts | Status |
|---|---|---|---|
| 1 | Naive → broken → fixed model | EDA, data leakage, sklearn pipelines | ✅ |
| 2 | Experiment tracking + registry | MLflow runs, aliases, comparison | ✅ |
| 3 | Structured + orchestrated pipeline | Clean code, retries, Prefect `@task`/`@flow` | ✅ |
| 4 | Online serving | FastAPI, Docker, MLflow aliases | ✅ |
| 5 | Batch scoring + drift monitoring | Async API, MAE ratio, Streamlit | ✅ |
| 6 | Full integrated local system | Docker Compose, service networking | ✅ |
| 7 | Running on a real VPS | SSH, firewall, remote MLflow, SSH tunnel | ✅ |
| 8 | Production hardening | Nginx, DuckDNS, SSL/Certbot, GitHub Actions CI/CD | ✅ |

### **The Moment This Module Earns Its Keep**

You train the model on 2019 taxi trips. You deploy it. Then you batch-score it forward through time, month by month — and in April 2020, its error rate more than doubles without a single line of code changing.

You don't need COVID explained to you. You lived it. That's the point: **this isn't a synthetic drift demo, it's a real system meeting a real discontinuity in the world**, and monitoring is the only reason anyone would have caught it before it cost something. Most students never *feel* why monitoring matters until it's too late on a real job. Here, you feel it in week 5.

Full module-by-module breakdown will live in that project's own README once its repo is public.

---

## 🛠️ **Technology Stack You'll Master**

<div align="center">

<table>
<tr>
<td align="center" width="20%">
<div style="background: #f3e5f5; padding: 15px; border-radius: 10px;">
<h4>🎯 Tracking</h4>
<code style="background: #f8f9fa; padding: 2px 6px; border-radius: 4px; display: block; margin: 5px 0;">📈 MLflow</code>
<code style="background: #f8f9fa; padding: 2px 6px; border-radius: 4px; display: block; margin: 5px 0;">🗂️ Model Registry</code>
</div>
</td>
<td align="center" width="20%">
<div style="background: #e3f2fd; padding: 15px; border-radius: 10px;">
<h4>🔄 Orchestration</h4>
<code style="background: #f8f9fa; padding: 2px 6px; border-radius: 4px; display: block; margin: 5px 0;">⚡ Prefect</code>
<code style="background: #f8f9fa; padding: 2px 6px; border-radius: 4px; display: block; margin: 5px 0;">🔁 Retries</code>
</div>
</td>
<td align="center" width="20%">
<div style="background: #fff3e0; padding: 15px; border-radius: 10px;">
<h4>🚀 Serving</h4>
<code style="background: #f8f9fa; padding: 2px 6px; border-radius: 4px; display: block; margin: 5px 0;">⚡ FastAPI</code>
<code style="background: #f8f9fa; padding: 2px 6px; border-radius: 4px; display: block; margin: 5px 0;">🐳 Docker</code>
</div>
</td>
<td align="center" width="20%">
<div style="background: #e8f5e9; padding: 15px; border-radius: 10px;">
<h4>📊 Monitoring</h4>
<code style="background: #f8f9fa; padding: 2px 6px; border-radius: 4px; display: block; margin: 5px 0;">🌊 Streamlit</code>
<code style="background: #f8f9fa; padding: 2px 6px; border-radius: 4px; display: block; margin: 5px 0;">📉 Drift Detection</code>
</div>
</td>
<td align="center" width="20%">
<div style="background: #fce4ec; padding: 15px; border-radius: 10px;">
<h4>☁️ Production</h4>
<code style="background: #f8f9fa; padding: 2px 6px; border-radius: 4px; display: block; margin: 5px 0;">🌐 Nginx</code>
<code style="background: #f8f9fa; padding: 2px 6px; border-radius: 4px; display: block; margin: 5px 0;">🔧 GitHub Actions</code>
</div>
</td>
</tr>
</table>

</div>

---

## 💼 **What You Walk Away Able to Do**

This is the difference between *"I've heard of Docker"* and being someone a hiring manager trusts with production systems on day one:

- **Debug a model that's silently degrading in production** — not just retrain it and hope, but diagnose *why* with drift metrics
- **Explain the tradeoffs of every tool you touch** — not recite what MLflow does, but say what breaks without it
- **Own a deployment end to end** — from `git push` to a live HTTPS endpoint with CI/CD, with no one holding your hand
- **Speak the language of an ML platform team in an interview** — because you've actually run the pager, not just read about it

We're not going to hand you invented statistics about job placement — this track is new, and we'd rather you trust real evidence than a marketing number. What we can promise is this: the gap between "trained a model in a notebook" and "operates a production ML system" is exactly the gap most junior candidates get filtered out on. This module closes it.

---

## 📖 **Required Reading**

<div align="center">
<a href="https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/">
<img src="https://covers.openlibrary.org/b/isbn/9781098107963-L.jpg" alt="Designing Machine Learning Systems by Chip Huyen — cover" width="180"/>
</a>
</div>

**[*Designing Machine Learning Systems*](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) by Chip Huyen** — required for Module 6, not optional.

Where First Principles and SAIRCAMP teach you *how to build and operate* one real system, this book is the missing layer above both: the design reasoning for *why* production ML systems end up shaped the way they do — data distribution shifts, train/serve skew, feature stores, monitoring strategy, and the org-level tradeoffs behind each decision. It gives you vocabulary and mental models for choices SAIRCAMP makes concretely (why a feature store, why batch vs. online scoring, why a model registry) instead of leaving you to reverse-engineer the reasoning from the code alone.

**How to read it for this module:** pair chapters with whatever you're currently building rather than reading cover to cover before starting — e.g. the monitoring/drift chapter around SAIRCAMP Module 5. You should have finished the book by the time you complete SAIRCAMP Project 1; being able to explain its concepts is part of what "completing" Module 6 means.

---

## 🎓 **Module 6 Completion Requirements**

<table>
<tr>
<td width="50%" valign="top">

**📚 Technical Excellence**
- ✅ Watch the full *MLOps from First Principles* playlist
- ✅ Complete SAIRCAMP Project 1 (Classical ML) end to end
- ✅ Read *Designing Machine Learning Systems* (Chip Huyen) and be able to discuss its concepts
- ✅ Reproduce the production deploy: Docker Compose → VPS → nginx + HTTPS + CI/CD
- ✅ Explain, for every tool used, what failure mode it exists to solve

</td>
<td width="50%" valign="top">

**🤝 Community & Professional Standards**
- ✅ Active participation in the live cohort sessions
- ✅ Present your deployed system with a live demo
- ✅ Push at least one real `git push origin main` that triggers CI/CD
- ✅ Help a peer debug their VPS or Docker setup

</td>
</tr>
</table>

> The 100 Days of MLOps track is the one optional piece here — recommended as proof of skill for your portfolio/interviews, but it is **not** a graded component of Module 6. Everything else on this page (both tracks, the book, the deploy) is required.

---

## 🚀 **Getting Started**

<div align="center">

<table>
<tr>
<td width="25%" align="center">
<h4><code>👥</code> 1️⃣ Join the Cohort</h4>
<a href="https://t.me/sair19969">Telegram: t.me/sair19969</a>
</td>
<td width="25%" align="center">
<h4><code>🎥</code> 2️⃣ Start the Theory Track</h4>
<a href="https://youtube.com/playlist?list=PLVM9Nqm8zLE0&si=jtIah3TJB8PjOMgu">MLOps from First Principles</a>
</td>
<td width="25%" align="center">
<h4><code>🏗️</code> 3️⃣ Follow the Live Builds</h4>
SAIRCAMP repos linked above,<br/>unlocked as each project releases
</td>
<td width="25%" align="center">
<h4><code>📅</code> 4️⃣ (Optional) Prove It</h4>
<a href="https://kodekloud.com/100-days-of-mlops">100 Days of MLOps</a><br/><sub>free, do it after 1-3, highly recommended</sub>
</td>
</tr>
</table>

</div>

```bash
# This repo is the map — clone it for the navigation and concept guides
git clone https://github.com/SAIR-Org/SAiR-MLOps-Blueprint.git

# Each SAIRCAMP project ships in its own repo once released, e.g.
git clone https://github.com/SAIR-Org/<saircamp-project-repo>.git
```

---

## 📚 **Where This Fits in SAIR Jr.**

```mermaid
flowchart TD
    M4["Module 4 - Applied Deep Learning<br/>SAIR-Org/SAIR_Jr"] --> M5["Module 5 - GPT from Scratch<br/>SAIR-Org/SAIR_Jr"]
    M5 --> M6["Module 6 - MLOps - you are here<br/>SAIR-Org/SAiR-MLOps-Blueprint"]

    M6 --> FP["MLOps from First Principles<br/>theory, required"]
    M6 --> SC["SAIRCAMP<br/>live builds, required"]
    M6 --> BK["Designing ML Systems<br/>Chip Huyen, required reading"]

    FP --> Cap["Capstone - Real-World Impact Project<br/>SAIR-Org/SAIR_Jr"]
    SC --> HD["100 Days of MLOps<br/>optional, recommended: proof of skill"]
    SC --> Cap
    BK --> Cap
    HD -.strengthens portfolio for.-> Cap

    classDef done fill:#238636,color:#fff
    classDef current fill:#d29922,color:#000
    classDef optional fill:#6e7681,color:#fff,stroke-dasharray: 5 5
    classDef next fill:#6e7681,color:#fff
    class M4,M5 done
    class M6,FP,SC,BK current
    class HD optional
    class Cap next
```

Module 6 is where everything from Modules 0-5 stops being *"a model in a notebook"* and becomes something a team could actually run. Head back to [SAIR_Jr](https://github.com/SAIR-Org/SAIR_Jr) for the rest of the track and the capstone.

---

<div align="center">

**License:** MIT | **Status:** 🔜 Incoming
**Building Sudan's AI Future, One Production System at a Time 🇸🇩✨**

</div>
