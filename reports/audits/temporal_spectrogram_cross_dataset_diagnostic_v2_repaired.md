# Cross-Dataset Temporal vs Spectrogram Diagnostic (Repaired Run)

**Pipeline valid:** `False`  
**Scientific verdict:** `INCONCLUSIVE`

This repaired run is separate from the prior provisional report. Raw caches are reused without modification; repaired predictions and reports use a new versioned output root.

## OBSERVED

This development-only report uses MATLAB Tr1/Va1 and VSB grouped development identities. VSB grouped test, official test, MATLAB Te1/Te2, CWT reruns, adaptive fusion, reliability models, and external weights remain locked.

### Temporal regression gates

| dataset | expected_mcc | expected_threshold | observed_mcc | observed_threshold | seed | status | tolerance_mcc | tolerance_threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vsb | 0.6288553457492239 | 0.265 | 0.6288553457492239 | 0.2649999999999999 | 42 | PASS | 1e-06 | 1e-12 |
| vsb | 0.6225752546362788 | 0.105 | 0.6225752546362788 | 0.10499999999999998 | 43 | PASS | 1e-06 | 1e-12 |
| vsb | 0.6512036719499463 | 0.195 | 0.6512036719499463 | 0.19499999999999995 | 44 | PASS | 1e-06 | 1e-12 |
| matlab | 0.9583519445260205 | 0.095 | 0.6081301062390931 | 0.065 | 42 | FAIL | 0.005 | 0.01 |
| matlab | 0.9699803927462562 | 0.15 | 0.8546432863984998 | 0.095 | 43 | FAIL | 0.005 | 0.01 |
| matlab | 0.971595033733952 | 0.13 | 0.9466987927743125 | 0.08499999999999999 | 44 | FAIL | 0.005 | 0.01 |

### Validation metrics

| dataset | seed | method | split | threshold | mcc | accuracy | precision | recall_pd | specificity | f1 | pr_auc | roc_auc | brier | ece_10_bins | predicted_positive_fraction | tn | fp | fn | tp | errors | temporal_weight |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| matlab | 42 | temporal | validation | 0.065 | 0.6081301062390931 | 0.7699782636491497 | 1.0 | 0.5399565272982995 | 1.0 | 0.701262039189638 | 0.8849032107651714 | 0.8458340606330681 | 0.2846268340880633 | 0.3066667471479945 | 0.2699782636491497 | 15642 | 0 | 7196 | 8446 | 7196 | nan |
| matlab | 42 | spectrogram | validation | 0.58 | 0.964446964168062 | 0.98190768443933 | 1.0 | 0.96381536887866 | 1.0 | 0.9815743212448728 | 0.9988924343255284 | 0.9982994019703852 | 0.0120626809221394 | 0.015140027667024 | 0.48190768443933 | 15642 | 0 | 566 | 15076 | 566 | nan |
| matlab | 43 | temporal | validation | 0.095 | 0.8546432863984998 | 0.9234752589182968 | 0.8734776725304465 | 0.9904104334484082 | 0.8565400843881856 | 0.9282761100125833 | 0.9941117685142542 | 0.99406755972453 | 0.0201480869909592 | 0.0311830902393833 | 0.5669351745301112 | 13398 | 2244 | 150 | 15492 | 2394 | nan |
| matlab | 43 | spectrogram | validation | 0.4149999999999999 | 0.946208642992736 | 0.9723820483314154 | 1.0 | 0.9447640966628308 | 1.0 | 0.9715976331360948 | 0.9994075388839404 | 0.9993207237092978 | 0.0239467314410109 | 0.0379145204326581 | 0.4723820483314154 | 15642 | 0 | 864 | 14778 | 864 | nan |
| matlab | 44 | temporal | validation | 0.0849999999999999 | 0.9466987927743123 | 0.9727017005498018 | 0.9987857528332432 | 0.9465541490857946 | 0.998849252013809 | 0.9719687520514672 | 0.9927180888985928 | 0.9895201482748156 | 0.0472397779594961 | 0.0538411943444299 | 0.4738524485359928 | 15624 | 18 | 836 | 14806 | 854 | nan |
| matlab | 44 | spectrogram | validation | 0.3599999999999999 | 0.9862111780981994 | 0.9930955120828538 | 0.996268176553854 | 0.98989898989899 | 0.996292034266718 | 0.9930733709594664 | 0.9996956021785932 | 0.999655064970938 | 0.006200160883979 | 0.0123402523937571 | 0.496803477816136 | 15584 | 58 | 158 | 15484 | 216 | nan |
| vsb | 42 | temporal | validation | 0.2649999999999999 | 0.6288553457492239 | 0.9586919104991394 | 0.67 | 0.6320754716981132 | 0.9798411728772144 | 0.6504854368932039 | 0.5928270470786552 | 0.9553486013300908 | 0.0346728542792892 | 0.0242714145967163 | 0.057372346528973 | 1604 | 33 | 39 | 67 | 72 | nan |
| vsb | 42 | spectrogram | validation | 0.6849999999999999 | 0.3196035534533086 | 0.9259896729776248 | 0.3789473684210526 | 0.3396226415094339 | 0.963958460598656 | 0.3582089552238806 | 0.2875040513174698 | 0.8806376136743467 | 0.2725572266439771 | 0.4732797557912335 | 0.0545037292025243 | 1578 | 59 | 70 | 36 | 129 | nan |
| vsb | 43 | temporal | validation | 0.1049999999999999 | 0.6225752546362788 | 0.9495123350545036 | 0.5642857142857143 | 0.7452830188679245 | 0.9627367135003054 | 0.6422764227642277 | 0.5928270470786552 | 0.9553486013300908 | 0.0346728542792892 | 0.0242714145967163 | 0.0803212851405622 | 1576 | 61 | 27 | 79 | 88 | nan |
| vsb | 43 | spectrogram | validation | 0.6649999999999999 | 0.2689181764482395 | 0.910499139414802 | 0.2950819672131147 | 0.3396226415094339 | 0.9474648747709224 | 0.3157894736842105 | 0.2435675014509673 | 0.8464344578785398 | 0.2728412445882801 | 0.4723745994846915 | 0.0699942627653471 | 1551 | 86 | 70 | 36 | 156 | nan |
| vsb | 44 | temporal | validation | 0.1949999999999999 | 0.6512036719499463 | 0.9586919104991394 | 0.6491228070175439 | 0.6981132075471698 | 0.9755650580329872 | 0.6727272727272727 | 0.5928270470786552 | 0.9553486013300908 | 0.0346728542792892 | 0.0242714145967163 | 0.0654044750430292 | 1597 | 40 | 32 | 74 | 72 | nan |
| vsb | 44 | spectrogram | validation | 0.63 | 0.4004763753502026 | 0.904188181296615 | 0.3368983957219251 | 0.5943396226415094 | 0.9242516799022602 | 0.4300341296928328 | 0.3305119069920714 | 0.8792602667096969 | 0.2736136844716766 | 0.4744699379184501 | 0.1072862880091795 | 1513 | 124 | 43 | 63 | 167 | nan |
| matlab | 42 | 50_50 | validation | 0.3149999999999999 | 0.9674765587667248 | 0.9834739803094232 | 1.0 | 0.9669479606188468 | 1.0 | 0.9831962817304256 | 0.9995374744160918 | 0.9994486009450588 | 0.082039861048988 | 0.1609033874075093 | 0.4834739803094233 | 15642 | 0 | 517 | 15125 | 517 | 0.5 |
| matlab | 42 | best_fixed | validation | 0.58 | 0.964446964168062 | 0.98190768443933 | 1.0 | 0.96381536887866 | 1.0 | 0.9815743212448728 | 0.9988924343255284 | 0.9982994019703852 | 0.0120626809221394 | 0.015140027667024 | 0.48190768443933 | 15642 | 0 | 566 | 15076 | 566 | 0.0 |
| matlab | 43 | 50_50 | validation | 0.2699999999999999 | 0.97423067667206 | 0.9870221199335124 | 0.9967396974439228 | 0.9772407620508886 | 0.996803477816136 | 0.9868939247207696 | 0.9988267409794396 | 0.9984732877091812 | 0.0177785456056127 | 0.0392151673086984 | 0.4902186421173763 | 15592 | 50 | 356 | 15286 | 406 | 0.5 |
| matlab | 43 | best_fixed | validation | 0.4149999999999999 | 0.946208642992736 | 0.9723820483314154 | 1.0 | 0.9447640966628308 | 1.0 | 0.9715976331360948 | 0.9994075388839404 | 0.9993207237092978 | 0.0239467314410109 | 0.0379145204326581 | 0.4723820483314154 | 15642 | 0 | 864 | 14778 | 864 | 0.0 |
| matlab | 44 | 50_50 | validation | 0.2699999999999999 | 0.9858684396078164 | 0.992903720751822 | 0.9984484096198604 | 0.9873417721518988 | 0.9984656693517452 | 0.9928640308582448 | 0.9997113332293124 | 0.999663872674948 | 0.0173012935094091 | 0.0390008911910742 | 0.4944380514000767 | 15618 | 24 | 198 | 15444 | 222 | 0.5 |
| matlab | 44 | best_fixed | validation | 0.3599999999999999 | 0.9862111780981994 | 0.9930955120828538 | 0.996268176553854 | 0.98989898989899 | 0.996292034266718 | 0.9930733709594664 | 0.9996956021785932 | 0.999655064970938 | 0.006200160883979 | 0.0123402523937571 | 0.496803477816136 | 15584 | 58 | 158 | 15484 | 216 | 0.0 |
| vsb | 42 | 50_50 | validation | 0.4049999999999999 | 0.5936767949266057 | 0.9512335054503728 | 0.5897435897435898 | 0.6509433962264151 | 0.9706780696395846 | 0.6188340807174888 | 0.5442649282975575 | 0.9432982561289058 | 0.0898782080898752 | 0.2356603501097511 | 0.0671256454388984 | 1589 | 48 | 37 | 69 | 85 | 0.5 |
| vsb | 42 | best_fixed | validation | 0.5599999999999999 | 0.5611206146931326 | 0.9500860585197934 | 0.5904761904761905 | 0.5849056603773585 | 0.9737324373854612 | 0.5876777251184834 | 0.4910708356930356 | 0.9304238079321354 | 0.1652835092739867 | 0.3561587642411507 | 0.0602409638554216 | 1594 | 43 | 44 | 62 | 87 | 0.25 |
| vsb | 43 | 50_50 | validation | 0.3699999999999999 | 0.6004817718442886 | 0.944348823866896 | 0.5302013422818792 | 0.7452830188679245 | 0.9572388515577276 | 0.6196078431372549 | 0.5270090633669068 | 0.9412696949090028 | 0.0899884831818135 | 0.2358313027486478 | 0.0854847963281698 | 1567 | 70 | 27 | 79 | 97 | 0.5 |
| vsb | 43 | best_fixed | validation | 0.5449999999999999 | 0.5292117083369676 | 0.9437751004016064 | 0.5344827586206896 | 0.5849056603773585 | 0.9670128283445328 | 0.5585585585585585 | 0.4710695291112013 | 0.9273175735641592 | 0.165472722322054 | 0.3553550741744293 | 0.0665519219736087 | 1583 | 54 | 44 | 62 | 98 | 0.25 |
| vsb | 44 | 50_50 | validation | 0.3999999999999999 | 0.5982371373087031 | 0.9506597819850832 | 0.5819672131147541 | 0.6698113207547169 | 0.9688454489920586 | 0.6228070175438597 | 0.5434720250116513 | 0.9457302244095848 | 0.0900961133375158 | 0.2362363992570375 | 0.0699942627653471 | 1586 | 51 | 35 | 71 | 86 | 0.5 |
| vsb | 44 | best_fixed | validation | 0.3999999999999999 | 0.5982371373087031 | 0.9506597819850832 | 0.5819672131147541 | 0.6698113207547169 | 0.9688454489920586 | 0.6228070175438597 | 0.5434720250116513 | 0.9457302244095848 | 0.0900961133375158 | 0.2362363992570375 | 0.0699942627653471 | 1586 | 51 | 35 | 71 | 86 | 0.5 |

### Prediction overlap and oracle

| dataset | seed | disagreement_rate | oracle_mcc | temporal_mcc | spectrogram_mcc | stronger_individual_mcc | oracle_headroom | group_count | stratum | state | count | fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| matlab | 42 | 0.2171077867280399 | 0.9694598606466888 | 0.6081301062390931 | 0.964446964168062 | 0.964446964168062 | 0.0050128964786265 | 31284.0 | nan | nan | nan | nan |
| matlab | 42 | nan | nan | nan | nan | nan | nan | nan | all | both_correct | 24007.0 | 0.7673890806802199 |
| matlab | 42 | nan | nan | nan | nan | nan | nan | nan | all | temporal_only | 81.0 | 0.0025891829689298 |
| matlab | 42 | nan | nan | nan | nan | nan | nan | nan | all | spectrogram_only | 6711.0 | 0.21451860375911 |
| matlab | 42 | nan | nan | nan | nan | nan | nan | nan | all | both_wrong | 485.0 | 0.0155031325917401 |
| matlab | 42 | nan | nan | nan | nan | nan | nan | nan | PD | both_correct | 8365.0 | 0.5347781613604399 |
| matlab | 42 | nan | nan | nan | nan | nan | nan | nan | PD | temporal_only | 81.0 | 0.0051783659378596 |
| matlab | 42 | nan | nan | nan | nan | nan | nan | nan | PD | spectrogram_only | 6711.0 | 0.4290372075182201 |
| matlab | 42 | nan | nan | nan | nan | nan | nan | nan | PD | both_wrong | 485.0 | 0.0310062651834803 |
| matlab | 42 | nan | nan | nan | nan | nan | nan | nan | NonPD | both_correct | 15642.0 | 1.0 |
| matlab | 42 | nan | nan | nan | nan | nan | nan | nan | NonPD | temporal_only | 0.0 | 0.0 |
| matlab | 42 | nan | nan | nan | nan | nan | nan | nan | NonPD | spectrogram_only | 0.0 | 0.0 |
| matlab | 42 | nan | nan | nan | nan | nan | nan | nan | NonPD | both_wrong | 0.0 | 0.0 |
| matlab | 43 | 0.0964071090653369 | 0.992294105829196 | 0.8546432863984998 | 0.946208642992736 | 0.946208642992736 | 0.0460854628364599 | 31284.0 | nan | nan | nan | nan |
| matlab | 43 | nan | nan | nan | nan | nan | nan | nan | all | both_correct | 28147.0 | 0.8997250990921877 |
| matlab | 43 | nan | nan | nan | nan | nan | nan | nan | all | temporal_only | 743.0 | 0.0237501598261091 |
| matlab | 43 | nan | nan | nan | nan | nan | nan | nan | all | spectrogram_only | 2273.0 | 0.0726569492392277 |
| matlab | 43 | nan | nan | nan | nan | nan | nan | nan | all | both_wrong | 121.0 | 0.0038677918424753 |
| matlab | 43 | nan | nan | nan | nan | nan | nan | nan | PD | both_correct | 14749.0 | 0.9429101137961896 |
| matlab | 43 | nan | nan | nan | nan | nan | nan | nan | PD | temporal_only | 743.0 | 0.0475003196522183 |
| matlab | 43 | nan | nan | nan | nan | nan | nan | nan | PD | spectrogram_only | 29.0 | 0.001853982866641 |
| matlab | 43 | nan | nan | nan | nan | nan | nan | nan | PD | both_wrong | 121.0 | 0.0077355836849507 |
| matlab | 43 | nan | nan | nan | nan | nan | nan | nan | NonPD | both_correct | 13398.0 | 0.8565400843881856 |
| matlab | 43 | nan | nan | nan | nan | nan | nan | nan | NonPD | temporal_only | 0.0 | 0.0 |
| matlab | 43 | nan | nan | nan | nan | nan | nan | nan | NonPD | spectrogram_only | 2244.0 | 0.1434599156118143 |
| matlab | 43 | nan | nan | nan | nan | nan | nan | nan | NonPD | both_wrong | 0.0 | 0.0 |
| matlab | 44 | 0.0257639688019434 | 0.9915954277743584 | 0.9466987927743123 | 0.9862111780981994 | 0.9862111780981994 | 0.005384249676159 | 31284.0 | nan | nan | nan | nan |
| matlab | 44 | nan | nan | nan | nan | nan | nan | nan | all | both_correct | 30346.0 | 0.970016621915356 |
| matlab | 44 | nan | nan | nan | nan | nan | nan | nan | all | temporal_only | 84.0 | 0.0026850786344457 |
| matlab | 44 | nan | nan | nan | nan | nan | nan | nan | all | spectrogram_only | 722.0 | 0.0230788901674977 |
| matlab | 44 | nan | nan | nan | nan | nan | nan | nan | all | both_wrong | 132.0 | 0.0042194092827004 |
| matlab | 44 | nan | nan | nan | nan | nan | nan | nan | PD | both_correct | 14779.0 | 0.9448280271065082 |
| matlab | 44 | nan | nan | nan | nan | nan | nan | nan | PD | temporal_only | 27.0 | 0.0017261219792865 |
| matlab | 44 | nan | nan | nan | nan | nan | nan | nan | PD | spectrogram_only | 705.0 | 0.0450709627924817 |
| matlab | 44 | nan | nan | nan | nan | nan | nan | nan | PD | both_wrong | 131.0 | 0.0083748881217235 |
| matlab | 44 | nan | nan | nan | nan | nan | nan | nan | NonPD | both_correct | 15567.0 | 0.995205216724204 |
| matlab | 44 | nan | nan | nan | nan | nan | nan | nan | NonPD | temporal_only | 57.0 | 0.0036440352896049 |
| matlab | 44 | nan | nan | nan | nan | nan | nan | nan | NonPD | spectrogram_only | 17.0 | 0.0010868175425137 |
| matlab | 44 | nan | nan | nan | nan | nan | nan | nan | NonPD | both_wrong | 1.0 | 6.393044367727913e-05 |
| vsb | 42 | 0.0453241537578886 | 0.6791530007510417 | 0.6288553457492239 | 0.3196035534533086 | 0.6288553457492239 | 0.0502976550018178 | 581.0 | nan | nan | nan | nan |
| vsb | 42 | nan | nan | nan | nan | nan | nan | nan | all | both_correct | 1603.0 | 0.9196787148594378 |
| vsb | 42 | nan | nan | nan | nan | nan | nan | nan | all | temporal_only | 68.0 | 0.0390131956397016 |
| vsb | 42 | nan | nan | nan | nan | nan | nan | nan | all | spectrogram_only | 11.0 | 0.006310958118187 |
| vsb | 42 | nan | nan | nan | nan | nan | nan | nan | all | both_wrong | 61.0 | 0.0349971313826735 |
| vsb | 42 | nan | nan | nan | nan | nan | nan | nan | PD | both_correct | 33.0 | 0.3113207547169811 |
| vsb | 42 | nan | nan | nan | nan | nan | nan | nan | PD | temporal_only | 34.0 | 0.320754716981132 |
| vsb | 42 | nan | nan | nan | nan | nan | nan | nan | PD | spectrogram_only | 3.0 | 0.0283018867924528 |
| vsb | 42 | nan | nan | nan | nan | nan | nan | nan | PD | both_wrong | 36.0 | 0.3396226415094339 |
| vsb | 42 | nan | nan | nan | nan | nan | nan | nan | NonPD | both_correct | 1570.0 | 0.9590714722052536 |
| vsb | 42 | nan | nan | nan | nan | nan | nan | nan | NonPD | temporal_only | 34.0 | 0.0207697006719609 |
| vsb | 42 | nan | nan | nan | nan | nan | nan | nan | NonPD | spectrogram_only | 8.0 | 0.0048869883934025 |
| vsb | 42 | nan | nan | nan | nan | nan | nan | nan | NonPD | both_wrong | 25.0 | 0.015271838729383 |
| vsb | 43 | 0.0780263912794033 | 0.7385912406156105 | 0.6225752546362788 | 0.2689181764482395 | 0.6225752546362788 | 0.1160159859793317 | 581.0 | nan | nan | nan | nan |
| vsb | 43 | nan | nan | nan | nan | nan | nan | nan | all | both_correct | 1553.0 | 0.8909925415949512 |
| vsb | 43 | nan | nan | nan | nan | nan | nan | nan | all | temporal_only | 102.0 | 0.0585197934595524 |
| vsb | 43 | nan | nan | nan | nan | nan | nan | nan | all | spectrogram_only | 34.0 | 0.0195065978198508 |
| vsb | 43 | nan | nan | nan | nan | nan | nan | nan | all | both_wrong | 54.0 | 0.0309810671256454 |
| vsb | 43 | nan | nan | nan | nan | nan | nan | nan | PD | both_correct | 32.0 | 0.3018867924528302 |
| vsb | 43 | nan | nan | nan | nan | nan | nan | nan | PD | temporal_only | 47.0 | 0.4433962264150943 |
| vsb | 43 | nan | nan | nan | nan | nan | nan | nan | PD | spectrogram_only | 4.0 | 0.0377358490566037 |
| vsb | 43 | nan | nan | nan | nan | nan | nan | nan | PD | both_wrong | 23.0 | 0.2169811320754717 |
| vsb | 43 | nan | nan | nan | nan | nan | nan | nan | NonPD | both_correct | 1521.0 | 0.9291386682956628 |
| vsb | 43 | nan | nan | nan | nan | nan | nan | nan | NonPD | temporal_only | 55.0 | 0.0335980452046426 |
| vsb | 43 | nan | nan | nan | nan | nan | nan | nan | NonPD | spectrogram_only | 30.0 | 0.0183262064752596 |
| vsb | 43 | nan | nan | nan | nan | nan | nan | nan | NonPD | both_wrong | 31.0 | 0.0189370800244349 |
| vsb | 44 | 0.0717154331612163 | 0.7201179326329075 | 0.6512036719499463 | 0.4004763753502026 | 0.6512036719499463 | 0.0689142606829611 | 581.0 | nan | nan | nan | nan |
| vsb | 44 | nan | nan | nan | nan | nan | nan | nan | all | both_correct | 1561.0 | 0.8955823293172691 |
| vsb | 44 | nan | nan | nan | nan | nan | nan | nan | all | temporal_only | 110.0 | 0.0631095811818703 |
| vsb | 44 | nan | nan | nan | nan | nan | nan | nan | all | spectrogram_only | 15.0 | 0.0086058519793459 |
| vsb | 44 | nan | nan | nan | nan | nan | nan | nan | all | both_wrong | 57.0 | 0.0327022375215146 |
| vsb | 44 | nan | nan | nan | nan | nan | nan | nan | PD | both_correct | 57.0 | 0.5377358490566038 |
| vsb | 44 | nan | nan | nan | nan | nan | nan | nan | PD | temporal_only | 17.0 | 0.160377358490566 |
| vsb | 44 | nan | nan | nan | nan | nan | nan | nan | PD | spectrogram_only | 6.0 | 0.0566037735849056 |
| vsb | 44 | nan | nan | nan | nan | nan | nan | nan | PD | both_wrong | 26.0 | 0.2452830188679245 |
| vsb | 44 | nan | nan | nan | nan | nan | nan | nan | NonPD | both_correct | 1504.0 | 0.9187538179596824 |
| vsb | 44 | nan | nan | nan | nan | nan | nan | nan | NonPD | temporal_only | 93.0 | 0.0568112400733048 |
| vsb | 44 | nan | nan | nan | nan | nan | nan | nan | NonPD | spectrogram_only | 9.0 | 0.0054978619425778 |
| vsb | 44 | nan | nan | nan | nan | nan | nan | nan | NonPD | both_wrong | 31.0 | 0.0189370800244349 |

### Hierarchical paired bootstrap

| bootstrap_mean | bootstrap_unit | ci_95 | comparison | dataset | fraction_gt_zero | iterations | point_estimate | seed_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1624660111909833 | individual_signal | [0.15937690505403407, 0.16563135967699033] | spectrogram_minus_temporal | matlab | 1.0 | 10000 | 0.16246486661569734 | 3 |
| 0.010237151857473838 | individual_signal | [0.009130181021913788, 0.011391670152911587] | 50_50_minus_best_individual | matlab | 1.0 | 10000 | 0.010236296595867925 | 3 |
| 0.0 | individual_signal | [0.0, 0.0] | best_fixed_minus_best_individual | matlab | 0.0 | 10000 | 0.0 | 3 |
| -0.30427672152517243 | nan | [-0.3657895441270491, -0.24479992323342306] | spectrogram_minus_temporal | vsb | 0.0 | 10000 | -0.3045453890278994 | 3 |
| -0.036784693498780316 | nan | [-0.0611079384735487, -0.013265461816305807] | 50_50_minus_best_individual | vsb | 0.0005 | 10000 | -0.03674618941861718 | 3 |
| -0.07138224157920516 | nan | [-0.10810769426361569, -0.036772961770117706] | best_fixed_minus_best_individual | vsb | 0.0001 | 10000 | -0.07135493733221525 | 3 |

### Additional diagnostics

- `threshold_curves.csv`: registered threshold grid for OOF and validation; selection is OOF-only.
- `probability_diagnostics.csv`: probability ranges and quantiles.
- `fixed_mixture_predictions.parquet`: diagnostic-only 50/50 and OOF-selected best-fixed mixtures.
- `vsb_aggregation_diagnostics.parquet`: event counts, top-k counts, event probabilities, half-cycle probabilities, and parent probabilities.

## INTERPRETATION

Pipeline validity and scientific verdict are reported separately. A failed temporal regression gate forces `INCONCLUSIVE`. Fixed mixtures are diagnostic-only and do not create an adaptive-fusion pipeline.

## UNRESOLVED

Dual-CyCon frequency concepts are verified only at high level. Erişti and exact reported literature values remain PROJECT LITERATURE ANCHOR — NOT REVERIFIED. Localization summaries are limited to information present in the immutable event cache.
