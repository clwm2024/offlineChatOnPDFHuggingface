# Ausführen von Frage-Antwort an eigene PDF Dokumente
**Stand:** 01.11.2024 <br>
**Autor:** Hans-Jörg Stark <br>
**Conda-Environment:** `langchanin_rag`

## Benötigte Module
Die benötigten Module sind in der Datei `requirements20241101.txt` aufgeführt und können im entsprechenden Conda Environment, das neu erstellt wurde mittels `conda create -n pdfchatter` mit dem Befehl `pip install -r requirements20241101.txt` installiert werden.

## Erstellung der Datenbank 
Die Datenbank wird erstellt, indem alle PDF Dokumente in Chunks geteilt werden.
Dies geschieht mit dem Skript `create_database_modified.py`. Im Skript ist auf Zeile 24 der Pfad zu den PDF Dokumenten anzugeben. Wird ein anderes Format verwendet, ist dies auf Zeile 36 zu ändern.

In der Datei `settings.env` ist der API Key für die Verwendung von OpenAI anzugeben. (Ist für die Erstellung der lokalen Datenbank NICHT nötig).

## Modell-Finetuning

### Erstellung einer Frage-Antwort Datei für das Finetuning
Mit Hilfe des Skripts `_createQuestion-Answer-sets_onHermes2.py` kann eine Datei im JSON Format erstellt werden, welche Pro Element 3 Unterelemente enthält: Kontext, Frage, Antwort. Diese Tripel dienen zum anschliessenden Finetuning. Diese Triplets werden in der Datei `questions_answers.json` gespeichert. Um das Skript erfolgreich ausführen zu können, ist ein separates Conda Environment mittels `environmentCreateQAFile4Finetuning.yaml` zu erstellen: `conda env create -f environmentCreateQAFile4Finetuning.yaml`. <br> 
Der Inhalt der Datei lautet:<br>
<pre><code>
name: openai1.0
channels:
  - conda-forge
  - defaults
dependencies:
  - bzip2=1.0.8=hfdf4475_7
  - ca-certificates=2024.8.30=h8857fd0_0
  - libexpat=2.6.3=hac325c4_0
  - libffi=3.4.2=h0d85af4_5
  - libsqlite=3.47.0=h2f8c449_1
  - libzlib=1.3.1=hd23fc13_2
  - ncurses=6.5=hf036a51_1
  - openssl=3.3.2=hd23fc13_0
  - pip=24.3.1=pyh8b19718_0
  - python=3.12.7=h8f8b54e_0_cpython
  - readline=8.2=h9e318b2_1
  - setuptools=75.3.0=pyhd8ed1ab_0
  - tk=8.6.13=h1abcd95_1
  - tzdata=2024b=hc8b5060_0
  - wheel=0.44.0=pyhd8ed1ab_0
  - xz=5.2.6=h775f41a_0
  - pip:
    - aiohappyeyeballs==2.4.3
    - aiohttp==3.10.10
    - aiosignal==1.3.1
    - annotated-types==0.7.0
    - anyio==4.6.2.post1
    - attrs==24.2.0
    - certifi==2024.8.30
    - charset-normalizer==3.4.0
    - distro==1.9.0
    - frozenlist==1.5.0
    - h11==0.14.0
    - httpcore==1.0.6
    - httpx==0.27.2
    - idna==3.10
    - jiter==0.7.0
    - multidict==6.1.0
    - openai==0.28.0
    - propcache==0.2.0
    - pydantic==2.9.2
    - pydantic-core==2.23.4
    - python-dotenv==1.0.1
    - requests==2.32.3
    - sniffio==1.3.1
    - tqdm==4.66.6
    - typing-extensions==4.12.2
    - urllib3==2.2.3
    - yarl==1.17.1
prefix: /opt/anaconda3/envs/openai1.0
</code></pre>


### Modell-Tuning 
Es gibt eine Datei mit Kontext, Fragen und Antworten: `questions_answers.json`. Diese wurde via OpenAI erstellt und kann zum Finetuning des Modells verwendet werden. Dazu ist ein separates Environment mit der Datei `environmentTrainModelElectra.yaml` zu erstellen: `conda env create -f environmentTrainModelElectra.yaml`. Der Inhalt der Datei lautet:<br>
<pre><code>
name: trainModelElectra
channels:
  - conda-forge
	- pytorch
  - defaults
dependencies:
  - pandas
  - pip
  - scikit-learn
  - numpy
  - python=3.10
  - pip:
      - transformers
      - datasets
      - evaluate
      - torch
      - torchvision
      - torchaudio
prefix: /opt/anaconda3/envs/trainModelElectra
</code></pre>

Anschliessend kann für das Finetuning die Datei `_trainGermanModelWithQandAFile.py` ausgeführt werden. <br> ACHTUNG: **Dieser Prozess kann sehr lange dauern!** <p>
Als Ergebnis liegt ein neues Modell vor im Verzeichnis `models/trainedGermanQAElectra` und kann anschliessend im Skript 


## Applikation starten
### 1. Im Browser mit Streamlit
Die Anwendung kann in der Konsole mit dem Befehl `streamlit run _runWithStreamlitExtendedHERMES.py` gestartet werden. Im Browser wird die Anwendung geöffnet und Fragen können gestellt werden. Die Abfragen gehen über die <u>*OpenAI API*</u> - also über das Internet!

### 2. Via Konsole
Im Terminal kann die Frage wie folgt gestellt werden: `python query_data_modified.py <FRAGE>` also konkret: `python query_data_modified.py "Wo kommt HERMES zum Einsatz?"`



