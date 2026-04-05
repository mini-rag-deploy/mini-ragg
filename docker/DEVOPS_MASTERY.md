# 🚀 Mastering the Backend & DevOps Stack (Mini-RAG)

هذا الملف مخصص لك لتتعلم وتفهم "الأساس العلمي والهندسي" (Under the Hood) من واقع **التوثيق الأصلي (Official Documentation)** لكل تقنية مستخدمة في المشروع. بدلاً من مجرد أخذ معلومات سطحية، إليك الخلاصة المعمارية لكل سيرفر لتصبح خبيراً في الـ Deployment.

---

## 1️⃣ تطبيق FastAPI & Uvicorn (The ASGI Web Framework)
**المرجع الأساسي:** [FastAPI Deployment Docs](https://fastapi.tiangolo.com/deployment/)

### المفهوم المعماري الرسمي (Architecture Concept):
- برنامجك المكتوب بالبايثون يحتاج إلى سيرفر يفهم لغة الإنترنت (HTTP) ويترجمها للبايثون والعكس. 
- **ASGI (Asynchronous Server Gateway Interface):** هو المعيار الحديث في البايثون للتعامل مع الطلبات المتزامنة (Asynchronous).
- **Uvicorn:** هو السيرفر الذي يطبق معيار ASGI. هو مجرد حلقة وصل، يستلم الطلب ويقول لـ FastAPI: "تفضل هذا الطلب، نفذ الكود الخاص بك، وأعطني الرد لأرسله للعميل".

### أهم مفاهيم يجب دراستها كمهندس خوادم:
* **Concurrency vs Parallelism:** الـ Uvicorn يعمل بنظام (Event Loop) واحد على نواة معالج (CPU Core) واحدة. لكي تستفيد من قوة معالج أو سيرفر بـ 8 أنوية (8 Cores)، يجب تشغيل Gunicorn كـ (Process Manager) ليقوم بتوليد 8 عمال (Workers) من Uvicorn، كل عامل يعمل على نواة مستقلة.
* **Statelessness:** يجب ألا تحفظ المتغيرات (Variables) في الذاكرة الحية للسيرفر لمشاركتها بين أكثر من طلب، لأن كل Worker لديه ذاكرة منفصلة. أي حفظ يجب أن يتم في قاعدة بيانات خارجية لتتمكن من إضافة (Scale Up) خوادم جديدة بلا مشاكل.

---

## 2️⃣ خادم Nginx (The Reverse Proxy)
**المرجع الأساسي:** [Nginx Architecture Docs](https://www.nginx.com/blog/inside-nginx-how-we-designed-for-performance-scale/)

### المفهوم المعماري الرسمي (Architecture Concept):
- الـ Nginx مصمم ليكون (Event-Driven) وغير معتمد على فتح Thread جديد لكل مستخدم (Block/Thread-per-connection). لذلك يمكنه تحمل آلاف الطلبات المتزامنة في نفس اللحظة بأقل استهلاك للرامات.
- يمتلك عملية رئيسية (Master Process) تدير عمليات فرعية (Worker Processes).

### أهم مفاهيم يجب دراستها كمهندس خوادم:
* **Reverse Proxy:** استلام الطلب من العميل (البروكسي العكسي)، التحدث نيابة عنه مع FastAPI، ثم إرجاع النتيجة للعميل، مما يخفي البنية التحتية الداخلية ويحميها.
* **Connection Pooling / Keepalive:** إبقاء الاتصالات مفتوحة بين Nginx و FastAPI لتجنب هدر وقت الـ (TCP Handshake) مع كل طلب جديد.
* **Rate Limiting & Load Balancing:** توزيع الحمل على أكثر من نسخة من التطبيق، ومنع السبام (Spam) وأي عميل يقوم بإرسال طلبات مكثفة (DDoS).

---

## 3️⃣ قاعدة PostgreSQL و إضافة PGVector
**المراجع:** [PostgreSQL Docs](https://www.postgresql.org/docs/) & [PGVector GitHub Repo](https://github.com/pgvector/pgvector)

### المفهوم المعماري الرسمي (Architecture Concept):
- **Postgres:** نظام إدارة قواعد بيانات علائقية قوي يعتمد على مبدأ صارم يسمى (ACID Compliance). يضمن لك عدم تلف البيانات حتى لو انقطع التيار الكهربائي أثناء الكتابة، بفضل الـ **WAL (Write-Ahead Logging)**، الذي يسجل العملية في قرص صلب (Disk) قبل إدخالها للذاكرة الحقيقية.
- **PGVector:** هي مجرد (Extension - إضافة) برمجية تمت إضافتها للبوستجرس، لتعطيه قدرة جديدة: فهم المصفوفات والأرقام العشرية (Vectors) واحتساب المسافات الرياضية بينها (مثل هندسة المتجهات).

### أهم مفاهيم يجب دراستها كمهندس خوادم:
* **Indexing (الفهرسة في Vector DB):** هناك أنواع فهارس. وأهمها **HNSW** (الرسوم البيانية المترابطة هرمياً). الفهرس يجعلك تبحث عن أقرب نصوص في مليسكوندز بدلاً من مسح ملايين السطور.
* **Distance Metrics:** يجب أن تفهم الطرق الرياضية لحساب التطابق: 
  * `Cosine Similarity` (تطابق الزوايا والمسار - ممتاز للغات).
  * `Euclidean` (المسافة الهندسية المباشرة).

---

## 4️⃣ محرك Qdrant (The Vector Search Engine)
**المرجع الأساسي:** [Qdrant Architecture](https://qdrant.tech/documentation/overview/architecture/)

### المفهوم المعماري الرسمي (Architecture Concept):
- هو محرك مبني باللغة العملاقة **Rust** (ما يجعله آمناً على الذاكرة وسريعاً جداً كسرعة الـ C++).
- مصمم خصيصاً للتعامل مع الـ Payload (المعلومات الوصفية Metadata) بجانب الـ Vector.
- يسمح بعمل (Filtering) صارم جداً مع البحث الموجه (Vector Search).

### أهم مفاهيم يجب دراستها كمهندس خوادم:
* **Storage Modes (أنماط التخزين):** في بيئة الإنتاج الكبيرة، محركات الـ Vectors تلتهم الرامات بصورة جنونية. توثيق Qdrant يقدم مفهوماً وهو الـ (Memmap) أو تخزين المتجهات على الـ (Disk) بدلاً من الرامات بشكل مباشر عند كبر حجمها لتوفير التكاليف.
* **Sharding & Replicas:** عندما تكبر الداتا عن قدرة سيرفر واحد، يقوم بتقسيم (Sharding) المحتوى على سيرفرات وتكرارها (Replication) من أجل السرعة في القراءة.

---

## 5️⃣ نظام المراقبة Prometheus (The Time-Series DB & Scraper)
**المرجع الأساسي:** [Prometheus Architecture](https://prometheus.io/docs/introduction/overview/)

### المفهوم المعماري الرسمي (Architecture Concept):
- هو نظام مراقبة وقاعدة بيانات من نوع (TSDB - Time Series Database)، مما يعني أن كل سطر يتم تخزينه يتكون من: `(زمن تسجيل المعلومة + المعلومة + الـ Labels الوصفية)`.
- **النمط الساحب (Pull Model):** عكس الأنظمة القديمة التي ترمي الأخطاء للوحة التحكم، Prometheus يذهب بنفسه عبر HTTP (كل 15 ثانية مثلاً) ليسأل سيرفراتك "اعطني أرقام استهلاكك الحالية"، وإذا سيرفر مات أو لم يرد، يسجل Prometheus أنه متعطل.

### أهم مفاهيم يجب دراستها كمهندس خوادم:
* **PromQL (Prometheus Query Language):** اللغة الأصلية للـ Prometheus للبحث. تستخدمها لعمل معادلات (Rate, Histogram, Sum) لمعرفة معدل الخطأ في الدقيقة أو متوسط سرعة استجابة الـ APIs.
* **Retention Policy:** تحديد مدة الاحتفاظ بالبيانات (مثلا 15 يوماً فقط) حتى لا يمتلئ القرص الصلب لديك.

---

## 6️⃣ لوحة القيادة Grafana (The Visualization Engine)
**المرجع الأساسي:** [Grafana Fundamentals](https://grafana.com/docs/grafana/latest/fundamentals/)

### المفهوم المعماري الرسمي (Architecture Concept):
- جرافانا لا يخزن البيانات (عدا إعدادات اللوحات). هو تطبيق يتصل بمصادر داتا (Data Sources) مثل (Prometheus أو Postgres أو MySQL)، ويأخذ منها الأرقام ثم يرسمها بسرعة فائقة على المتصفح.

### أهم مفاهيم يجب دراستها كمهندس خوادم:
* **Dashboards as Code (Provisioning):** في بيئة العمل الحقيقية لا يتم تصميم اللوحة يدوياً (بالماوس)، بل تكتب كملف JSON (Dashboard Provisioning) يقوم الـ Docker بحقنها تلقائياً للمشروع لضمان ثبات الإعدادات بين سيرفر التطوير وسيرفر الإنتاج (Production).
* **Alerting (نظام الإنذارات):** ربط جرافانا بـ Slack أو Email ليرسل رسالة للمهندس آلياً إذا تجاوز استهلاك الـ CPU %90 أو إذا ارتفع معدل خطأ (HTTP 500) عن وتيرة معينة.

---

## 7️⃣ وسطاء سحب البيانات (Exporters / Instrumentation)
**المرجع الأساسي:** [Prometheus Exporters Concept](https://prometheus.io/docs/instrumenting/exporters/)

### المفهوم المعماري الرسمي (Architecture Concept):
- الـ Prometheus يفهم لغة ونصوصاً (Format) معينة. أي نظام أساسي لا يمتلك زر إخراج لهذه النصوص سيتطلب وسيطاً (Exporter).
- الـ Exporter هو سكربت/تطبيق خفيف جداً يختبئ بجوار النظام الأساسي، يتحدث لغته، يترجم حالته لصيغة `key=value` البسيطة، ثم يفتح منفذاً (`/metrics`) ليأتي Prometheus ويسحب منه النص.

### أهم مفاهيم يجب دراستها كمهندس خوادم:
* **Node Exporter:** يفهم ملفات نواة لينكس (مثل: `/proc/` و `/sys/`) ليجلب استهلاك الأنوية، سرعة نقل الشبكة، والقرص الصلب.
* **Postgres Exporter:** يدخل على جداول البوستجرس المخفية (مثل جدول `pg_stat_activity` المهتم بالاتصالات الحالية) ويترجم لك: كم اتصال نشط حالياً، وكم استعلام بطيء یتجاوز الـ 5 ثواني.