# maturaKWI2020
Repository for my Matura project on recognizing bird species using CNNs.

Willkommen im GitHub-Repositorium meiner Maturitätsarbeit. Hier finden sie (fast) alle Dateien exklusive des Lernsets, welche ich für mein Projekt verwendet oder erstellt habe. Sämtliche Dateinamen, Beschreibungen und Kommentare sind auf Englisch, da der Python-Syntax stark der englischen Sprache ähnelt. Dazu wird auch im deutschsprachigen Raum in der Informatik vermehrt auf Englisch gesetzt.

Die wichtigsten Dateien im Repositorium sind mobilenet_learning und detect-classifier. Beide sind Python-Dateien, welche den Code für den Lernskript beziehungsweise für den Testskript enthalten. Falls sie den detect-classifier selber ausprobieren möchten, sollten sie sichergehen, dass sie das neueste Python mit allen nötigen Bibliotheken installiert haben. Zusätzlich brauchen sie labelbin_2020-09-20 aus dem Ordner labelbins und def_2020-10-10 aus dem Ordner models. Wenn sie dabei sind, den Skript im Terminal auszuführen, sollten sie folgendes beachten: Sie müssen das Modell und den LabelBinarizer als Argumente im Terminal eingeben. Das würde etwa so aussehen: python detect-classifier.py -m [MODEL PATH] -l [LABELBINARIZER PATH] (-etc). 