# Gestufter Fit der Paper-SFG-Fingerprints an SiO2/Wasser

## Methode

- Die magenta Total-Kurven wurden direkt aus den EPS-Dateien des eigenen Papers digitisiert; sie wurden nicht analytisch neu gezeichnet.
- Gefittet wurde nur 3300-3800 cm^-1, also der OH-Streckbereich. Die starken Strukturen unter 3000 cm^-1 werden hier nicht als Wasserorientierung verwendet.
- Jede aktive Struktur bekommt ein nichtnegatives Gewicht; wenn ein Motiv in einem Modell genannt ist, muss es im Fit positiv populiert sein.
- Ein konstanter plus linearer Hintergrund ist immer enthalten.
- Fuer jedes Modell wurden ein globaler Frequenzshift und ein globales Vorzeichen getestet. Das Vorzeichen ist eine Phasen-/Normalenkonvention der Dateien; physikalisch wird die Cyran-Konvention benutzt: positives Im(chi) bedeutet H/OH netto in Richtung SiO2.
- Orientierung: Was im Paper nach oben zeigt, wird hier als Richtung SiO2/Fenster gelesen.
- p-Werte sind F-Test-Diagnostik gegen den linearen Hintergrund bzw. in der Stufentabelle gegen die vorherige Stufe. Wegen Nichtnegativitaet, digitisierten Kurven und reoptimiertem Shift sind sie als Modellvergleich zu lesen, nicht als strenger experimenteller Signifikanztest.

## Stufenmodell

| Stufe | aktive Spezies | R2 | Shift / Vorzeichen | p gegen Hintergrund | Delta SSE zur Vorstufe | p zur Vorstufe |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 minimal I+L2 | I, VII, VIII | 0.6512 | -37.0 / -1 | 3.425e-82 |  |  |
| 1 + II/VI bridge | I, VI, VII, VIII | 0.6442 | -35.0 / -1 | 1.193e-78 | -0.000446 |  |
| 2 + IV parallel | I, VI, VII, VIII | 0.6442 | -35.0 / -1 | 1.193e-78 | 0 |  |
| 3 + III/V | I, VI, VII, VIII | 0.6442 | -35.0 / -1 | 1.193e-78 | 0 |  |

## Minimalmodell I + L2

- R2 Minimalmodell: 0.6512
- SSE Minimalmodell: 0.0221687
- Aktive Spezies: I, VII, VIII
- Bestes BIC innerhalb der gestuften Modelle: minimal_l1a_l2 (I, VII, VIII)
- Bestes R2 innerhalb der gestuften Modelle: minimal_l1a_l2 (I, VII, VIII)

| species | region | rms fraction | structural note |
| --- | --- | ---: | --- |
| VIII | L2 | 0.507 | second-layer H-bonded motif, partially cancels VII |
| VII | L2 | 0.493 | second-layer H-bonded motif, opposite orientation to VIII |
| I | L1 Region A | 0.000 | one nearly free OH plus one OH directed back to the H-bond network |

## Flexible L1B-Kontrolle

- Als Kontrollfit wurde auch eine freie, aber schichtbesetzte L1B-Mischung gerechnet: I ist gesetzt, mindestens eine Spezies aus II-VI ist vorhanden, und L2 enthaelt VII und/oder VIII.
- R2 flexible L1B-Kontrolle: 0.6442
- Aktive Spezies flexible L1B-Kontrolle: I, VI, VII, VIII
- R2 komplett unbeschraenkter Fingerprintfit: 0.6774; aktive Spezies: VII, VIII

| species | region | rms fraction | structural note |
| --- | --- | ---: | --- |
| VIII | L2 | 0.518 | second-layer H-bonded motif, partially cancels VII |
| VII | L2 | 0.451 | second-layer H-bonded motif, opposite orientation to VIII |
| VI | L1 Region B | 0.022 | connector from Region B toward L2; mostly H-bonded with weakly bonded shoulder |
| I | L1 Region A | 0.009 | one nearly free OH plus one OH directed back to the H-bond network |

## Strukturinterpretation

- Die Tiefeninformation kommt aus Fig. 1 des Papers: SiO2-Seite -> L1/Region A -> L1/Region B -> L2 -> tieferes Wasser. Es werden keine harten 3-Angstrom-Slabs angenommen.
- Das Minimalmodell koppelt L1A/species I mit L2/species VII/VIII. Strukturell ist das der kleinste Wasser-only Ansatz, in dem die SiO2-seitige schwach/frei-OH-artige Orientierung und ein tieferes H-Brueckennetz gleichzeitig vorhanden sind.
- II und VI sind der erste sinnvolle L1B-Zusatz, weil II an die L1A-Seite und VI an L2 koppeln kann. IV testet danach die parallel orientierte, lokal schwache aber kopplungsaktive Population. III/V testen zuletzt weitere Region-B-H-Brueckenvarianten.
- Wenn II und VI beide zwingend erzwungen werden, faellt R2 auf 0.6350. Der II/VI-Pool-Fit waehlt deshalb nur die spektral getragene Brueckenkomponente.
- Die Gewichte sind spektrale Fingerprint-Gewichte, keine direkten Molekuelzahlen.
- Frequenzen unter 3000 cm^-1 bleiben ausserhalb dieses Wasserstrukturmodells; wenn sie im Rohspektrum dominieren, sind sie fuer diese Wasser-OH-Strukturzuordnung nicht belastbar.