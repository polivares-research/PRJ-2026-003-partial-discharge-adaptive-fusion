# Cross-Dataset Temporal vs Spectrogram Diagnostic

**Verdict:** `SPECTROGRAM SUPPORTED ONLY ON MATLAB`

## OBSERVED

This development-only report uses MATLAB Tr1/Va1 and VSB grouped development identities. VSB grouped test, official test, MATLAB Te1/Te2, CWT reruns, and adaptive fusion were not opened.

### Validation metrics

| dataset | seed | method | threshold | mcc | accuracy | precision | recall_pd | specificity | f1 | pr_auc | roc_auc | brier | ece_10_bins | tn | fp | fn | tp | errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| matlab | 42 | spectrogram | 0.3649999999999999 | 0.9782870487547444 | 0.989099859353024 | 0.9826509368494102 | 0.9957805907172996 | 0.9824191279887482 | 0.989172196996158 | 0.999677886553105 | 0.9996547053059948 | 0.0073136328847725 | 0.0174484645640045 | 15367 | 275 | 66 | 15576 | 341 |
| matlab | 42 | temporal | 0.2049999999999999 | 0.9523653722939328 | 0.975962153177343 | 0.9909006989318212 | 0.9607467075821506 | 0.9911775987725356 | 0.9755907556478836 | 0.995177066939749 | 0.9935377303484348 | 0.0287362957581492 | 0.0404077902229956 | 15504 | 138 | 614 | 15028 | 752 |
| matlab | 43 | spectrogram | 0.5499999999999999 | 0.9832459514271324 | 0.9915611814345991 | 0.999480317006626 | 0.9836338064186164 | 0.9994885564505818 | 0.991493749194484 | 0.999658751843584 | 0.9996147906714882 | 0.0060461548713616 | 0.0090726817241706 | 15634 | 8 | 256 | 15386 | 264 |
| matlab | 43 | temporal | 0.095 | 0.8546432863984998 | 0.9234752589182968 | 0.8734776725304465 | 0.9904104334484082 | 0.8565400843881856 | 0.9282761100125833 | 0.9941117685142542 | 0.99406755972453 | 0.0201480869909592 | 0.0311830902393833 | 13398 | 2244 | 150 | 15492 | 2394 |
| matlab | 44 | spectrogram | 0.3699999999999999 | 0.985551719728935 | 0.9927758598644676 | 0.9927758598644676 | 0.9927758598644676 | 0.9927758598644676 | 0.9927758598644676 | 0.999698215106751 | 0.9996705142150948 | 0.0057565710718639 | 0.0094713934793288 | 15529 | 113 | 113 | 15529 | 226 |
| matlab | 44 | temporal | 0.1549999999999999 | 0.9710409035524644 | 0.9854238588415803 | 0.995303326810176 | 0.9754507096279248 | 0.995397008055236 | 0.9852770244091438 | 0.9979576671653124 | 0.9973041109817464 | 0.0192442960928278 | 0.017694070734196 | 15570 | 72 | 384 | 15258 | 456 |
| vsb | 42 | spectrogram | 0.5399999999999999 | -0.0408902511813615 | 0.7986230636833046 | 0.0377358490566037 | 0.0943396226415094 | 0.8442272449602932 | 0.0539083557951482 | 0.0826198649160871 | 0.6343259068014431 | 0.2658148090438295 | 0.4554048263259776 | 1382 | 255 | 96 | 10 | 351 |
| vsb | 42 | temporal | 0.2649999999999999 | 0.6288553457492239 | 0.9586919104991394 | 0.67 | 0.6320754716981132 | 0.9798411728772144 | 0.6504854368932039 | 0.5928270470786552 | 0.9553486013300908 | 0.0346728542792891 | 0.0242714145967164 | 1604 | 33 | 39 | 67 | 72 |
| vsb | 43 | spectrogram | 0.565 | 0.1327769612944182 | 0.2748135398737808 | 0.0773722627737226 | 1.0 | 0.2278558338423946 | 0.1436314363143631 | 0.1337642436327853 | 0.7434561611784096 | 0.3539212767306399 | 0.5466180301708312 | 373 | 1264 | 0 | 106 | 1264 |
| vsb | 43 | temporal | 0.1049999999999999 | 0.6225752546362788 | 0.9495123350545036 | 0.5642857142857143 | 0.7452830188679245 | 0.9627367135003054 | 0.6422764227642277 | 0.5928270470786552 | 0.9553486013300908 | 0.0346728542792891 | 0.0242714145967164 | 1576 | 61 | 27 | 79 | 88 |
| vsb | 44 | spectrogram | 0.575 | 0.0 | 0.9391853126792886 | 0.0 | 0.0 | 1.0 | 0.0 | 0.1267370863537849 | 0.7626871520614101 | 0.2503563265417265 | 0.4396112649282704 | 1637 | 0 | 106 | 0 | 106 |
| vsb | 44 | temporal | 0.1949999999999999 | 0.6512036719499463 | 0.9586919104991394 | 0.6491228070175439 | 0.6981132075471698 | 0.9755650580329872 | 0.6727272727272727 | 0.5928270470786552 | 0.9553486013300908 | 0.0346728542792891 | 0.0242714145967164 | 1597 | 40 | 32 | 74 | 72 |

### Prediction overlap

| dataset | seed | disagreement_rate | oracle_mcc | temporal_mcc | spectrogram_mcc | stronger_individual_mcc | oracle_headroom | group_count | stratum | state | count | fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| matlab | 42 | 0.0309103695179644 | 0.9959776759583078 | 0.9523653722939328 | 0.9782870487547444 | 0.9782870487547444 | 0.0176906272035632 | 31284.0 | nan | nan | nan | nan |
| matlab | 42 | nan | nan | nan | nan | nan | nan | nan | all | both_correct | 30254.0 | 0.9670758215062012 |
| matlab | 42 | nan | nan | nan | nan | nan | nan | nan | all | temporal_only | 278.0 | 0.0088863316711417 |
| matlab | 42 | nan | nan | nan | nan | nan | nan | nan | all | spectrogram_only | 689.0 | 0.0220240378468226 |
| matlab | 42 | nan | nan | nan | nan | nan | nan | nan | all | both_wrong | 63.0 | 0.0020138089758342 |
| matlab | 42 | nan | nan | nan | nan | nan | nan | nan | PD | both_correct | 15019.0 | 0.9601713335890552 |
| matlab | 42 | nan | nan | nan | nan | nan | nan | nan | PD | temporal_only | 9.0 | 0.0005753739930955 |
| matlab | 42 | nan | nan | nan | nan | nan | nan | nan | PD | spectrogram_only | 557.0 | 0.0356092571282444 |
| matlab | 42 | nan | nan | nan | nan | nan | nan | nan | PD | both_wrong | 57.0 | 0.0036440352896049 |
| matlab | 42 | nan | nan | nan | nan | nan | nan | nan | NonPD | both_correct | 15235.0 | 0.9739803094233472 |
| matlab | 42 | nan | nan | nan | nan | nan | nan | nan | NonPD | temporal_only | 269.0 | 0.017197289349188 |
| matlab | 42 | nan | nan | nan | nan | nan | nan | nan | NonPD | spectrogram_only | 132.0 | 0.0084388185654008 |
| matlab | 42 | nan | nan | nan | nan | nan | nan | nan | NonPD | both_wrong | 6.0 | 0.0003835826620636 |
| matlab | 43 | 0.0804244981460171 | 0.9954706238235912 | 0.8546432863984998 | 0.9832459514271324 | 0.9832459514271324 | 0.0122246723964588 | 31284.0 | nan | nan | nan | nan |
| matlab | 43 | nan | nan | nan | nan | nan | nan | nan | all | both_correct | 28697.0 | 0.9173059711034396 |
| matlab | 43 | nan | nan | nan | nan | nan | nan | nan | all | temporal_only | 193.0 | 0.0061692878148574 |
| matlab | 43 | nan | nan | nan | nan | nan | nan | nan | all | spectrogram_only | 2323.0 | 0.0742552103311597 |
| matlab | 43 | nan | nan | nan | nan | nan | nan | nan | all | both_wrong | 71.0 | 0.0022695307505434 |
| matlab | 43 | nan | nan | nan | nan | nan | nan | nan | PD | both_correct | 15306.0 | 0.9785193709244342 |
| matlab | 43 | nan | nan | nan | nan | nan | nan | nan | PD | temporal_only | 186.0 | 0.0118910625239739 |
| matlab | 43 | nan | nan | nan | nan | nan | nan | nan | PD | spectrogram_only | 80.0 | 0.0051144354941823 |
| matlab | 43 | nan | nan | nan | nan | nan | nan | nan | PD | both_wrong | 70.0 | 0.0044751310574095 |
| matlab | 43 | nan | nan | nan | nan | nan | nan | nan | NonPD | both_correct | 13391.0 | 0.8560925712824446 |
| matlab | 43 | nan | nan | nan | nan | nan | nan | nan | NonPD | temporal_only | 7.0 | 0.0004475131057409 |
| matlab | 43 | nan | nan | nan | nan | nan | nan | nan | NonPD | spectrogram_only | 2243.0 | 0.143395985168137 |
| matlab | 43 | nan | nan | nan | nan | nan | nan | nan | NonPD | both_wrong | 1.0 | 6.393044367727913e-05 |
| matlab | 44 | 0.0166858457997698 | 0.9948949656897976 | 0.9710409035524644 | 0.985551719728935 | 0.985551719728935 | 0.0093432459608625 | 31284.0 | nan | nan | nan | nan |
| matlab | 44 | nan | nan | nan | nan | nan | nan | nan | all | both_correct | 30682.0 | 0.980756936453139 |
| matlab | 44 | nan | nan | nan | nan | nan | nan | nan | all | temporal_only | 146.0 | 0.0046669223884413 |
| matlab | 44 | nan | nan | nan | nan | nan | nan | nan | all | spectrogram_only | 376.0 | 0.0120189234113284 |
| matlab | 44 | nan | nan | nan | nan | nan | nan | nan | all | both_wrong | 80.0 | 0.0025572177470911 |
| matlab | 44 | nan | nan | nan | nan | nan | nan | nan | PD | both_correct | 15219.0 | 0.9729574223245108 |
| matlab | 44 | nan | nan | nan | nan | nan | nan | nan | PD | temporal_only | 39.0 | 0.0024932873034138 |
| matlab | 44 | nan | nan | nan | nan | nan | nan | nan | PD | spectrogram_only | 310.0 | 0.0198184375399565 |
| matlab | 44 | nan | nan | nan | nan | nan | nan | nan | PD | both_wrong | 74.0 | 0.0047308528321186 |
| matlab | 44 | nan | nan | nan | nan | nan | nan | nan | NonPD | both_correct | 15463.0 | 0.988556450581767 |
| matlab | 44 | nan | nan | nan | nan | nan | nan | nan | NonPD | temporal_only | 107.0 | 0.0068405574734688 |
| matlab | 44 | nan | nan | nan | nan | nan | nan | nan | NonPD | spectrogram_only | 66.0 | 0.0042194092827004 |
| matlab | 44 | nan | nan | nan | nan | nan | nan | nan | NonPD | both_wrong | 6.0 | 0.0003835826620636 |
| vsb | 42 | 0.1715433161216293 | 0.6674610098443132 | 0.6288553457492239 | -0.0408902511813615 | 0.6288553457492239 | 0.0386056640950893 | 581.0 | nan | nan | nan | nan |
| vsb | 42 | nan | nan | nan | nan | nan | nan | nan | all | both_correct | 1382.0 | 0.7928858290304074 |
| vsb | 42 | nan | nan | nan | nan | nan | nan | nan | all | temporal_only | 289.0 | 0.165806081468732 |
| vsb | 42 | nan | nan | nan | nan | nan | nan | nan | all | spectrogram_only | 10.0 | 0.0057372346528973 |
| vsb | 42 | nan | nan | nan | nan | nan | nan | nan | all | both_wrong | 62.0 | 0.0355708548479632 |
| vsb | 42 | nan | nan | nan | nan | nan | nan | nan | PD | both_correct | 10.0 | 0.0943396226415094 |
| vsb | 42 | nan | nan | nan | nan | nan | nan | nan | PD | temporal_only | 57.0 | 0.5377358490566038 |
| vsb | 42 | nan | nan | nan | nan | nan | nan | nan | PD | spectrogram_only | 0.0 | 0.0 |
| vsb | 42 | nan | nan | nan | nan | nan | nan | nan | PD | both_wrong | 39.0 | 0.3679245283018867 |
| vsb | 42 | nan | nan | nan | nan | nan | nan | nan | NonPD | both_correct | 1372.0 | 0.83811850946854 |
| vsb | 42 | nan | nan | nan | nan | nan | nan | nan | NonPD | temporal_only | 232.0 | 0.1417226634086744 |
| vsb | 42 | nan | nan | nan | nan | nan | nan | nan | NonPD | spectrogram_only | 10.0 | 0.0061087354917532 |
| vsb | 42 | nan | nan | nan | nan | nan | nan | nan | NonPD | both_wrong | 23.0 | 0.0140500916310323 |
| vsb | 43 | 0.7056798623063684 | 0.7817150331164124 | 0.6225752546362788 | 0.1327769612944182 | 0.6225752546362788 | 0.1591397784801336 | 581.0 | nan | nan | nan | nan |
| vsb | 43 | nan | nan | nan | nan | nan | nan | nan | all | both_correct | 452.0 | 0.2593230063109581 |
| vsb | 43 | nan | nan | nan | nan | nan | nan | nan | all | temporal_only | 1203.0 | 0.6901893287435457 |
| vsb | 43 | nan | nan | nan | nan | nan | nan | nan | all | spectrogram_only | 27.0 | 0.0154905335628227 |
| vsb | 43 | nan | nan | nan | nan | nan | nan | nan | all | both_wrong | 61.0 | 0.0349971313826735 |
| vsb | 43 | nan | nan | nan | nan | nan | nan | nan | PD | both_correct | 79.0 | 0.7452830188679245 |
| vsb | 43 | nan | nan | nan | nan | nan | nan | nan | PD | temporal_only | 0.0 | 0.0 |
| vsb | 43 | nan | nan | nan | nan | nan | nan | nan | PD | spectrogram_only | 27.0 | 0.2547169811320754 |
| vsb | 43 | nan | nan | nan | nan | nan | nan | nan | PD | both_wrong | 0.0 | 0.0 |
| vsb | 43 | nan | nan | nan | nan | nan | nan | nan | NonPD | both_correct | 373.0 | 0.2278558338423946 |
| vsb | 43 | nan | nan | nan | nan | nan | nan | nan | NonPD | temporal_only | 1203.0 | 0.7348808796579108 |
| vsb | 43 | nan | nan | nan | nan | nan | nan | nan | NonPD | spectrogram_only | 0.0 | 0.0 |
| vsb | 43 | nan | nan | nan | nan | nan | nan | nan | NonPD | both_wrong | 61.0 | 0.0372632864996945 |
| vsb | 44 | 0.0654044750430292 | 0.8274830348778247 | 0.6512036719499463 | 0.0 | 0.6512036719499463 | 0.1762793629278783 | 581.0 | nan | nan | nan | nan |
| vsb | 44 | nan | nan | nan | nan | nan | nan | nan | all | both_correct | 1597.0 | 0.9162363740676994 |
| vsb | 44 | nan | nan | nan | nan | nan | nan | nan | all | temporal_only | 74.0 | 0.04245553643144 |
| vsb | 44 | nan | nan | nan | nan | nan | nan | nan | all | spectrogram_only | 40.0 | 0.0229489386115892 |
| vsb | 44 | nan | nan | nan | nan | nan | nan | nan | all | both_wrong | 32.0 | 0.0183591508892713 |
| vsb | 44 | nan | nan | nan | nan | nan | nan | nan | PD | both_correct | 0.0 | 0.0 |
| vsb | 44 | nan | nan | nan | nan | nan | nan | nan | PD | temporal_only | 74.0 | 0.6981132075471698 |
| vsb | 44 | nan | nan | nan | nan | nan | nan | nan | PD | spectrogram_only | 0.0 | 0.0 |
| vsb | 44 | nan | nan | nan | nan | nan | nan | nan | PD | both_wrong | 32.0 | 0.3018867924528302 |
| vsb | 44 | nan | nan | nan | nan | nan | nan | nan | NonPD | both_correct | 1597.0 | 0.9755650580329872 |
| vsb | 44 | nan | nan | nan | nan | nan | nan | nan | NonPD | temporal_only | 0.0 | 0.0 |
| vsb | 44 | nan | nan | nan | nan | nan | nan | nan | NonPD | spectrogram_only | 40.0 | 0.0244349419670128 |
| vsb | 44 | nan | nan | nan | nan | nan | nan | nan | NonPD | both_wrong | 0.0 | 0.0 |

## INTERPRETATION

The verdict is registered only after integrity, split, cache, alignment, and temporal regression checks. Fixed mixtures are diagnostic-only.

## UNRESOLVED

Dual-CyCon frequency concepts are verified only at high level. Erişti and exact reported literature values remain PROJECT LITERATURE ANCHOR — NOT REVERIFIED.
