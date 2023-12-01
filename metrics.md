Trained in GPT texts
- Composition questions accuracy: 0.9807661864809245
- Structure questions accuracy: 0.2609624821909134  # 반대로 해서 다시 해야함 
- Metal questions accuracy: 0.5739274972296976
- Semiconductor questions accuracy: 0.4738800063321197
- Stable questions accuracy: 0.471663764445148  # 반대로 해서 다시해야 함
- Oxide questions accuracy: 0.8762862118094032

Trained in academic paper paragraphs
- Composition questions accuracy: 0.849058097198037
- Structure questions accuracy: 0.16511002057938895
- Metal questions accuracy: 0.5249327212284313
- Semiconductor questions accuracy: 0.45987019154662023
- Stable questions accuracy: 0.4426943169225898
- Oxide questions accuracy: 0.8337027069811619

Trained in GPT texts - painn
- Composition questions accuracy: 0.9943802437866076
- Structure questions accuracy: 0.18244419819534588  # 반대로 해서 다시 해야함 
- Metal questions accuracy: 0.3043375019787874
- Semiconductor questions accuracy: 0.4762545512110179
- Stable questions accuracy: 0.697799588412221  # 반대로 해서 다시해야 함
- Oxide questions accuracy: 0.9205318980528732

Trained in academic paper paragraphs - painn
- Composition questions accuracy: 0.8947285103688459
- Structure questions accuracy: 0.16875098939369954
- Metal questions accuracy: 0.5917365838214342
- Semiconductor questions accuracy: 0.5752730726610733
- Stable questions accuracy: 0.5347475067278772
- Oxide questions accuracy: 0.6466677220199462

Trained in GPT texts - painn - galax
- Composition questions accuracy: 0.9123001424726928
- Structure questions accuracy: 0.20729776792781385  # 반대로 해서 다시 해야함 
- Metal questions accuracy: 0.2574006648725661
- Semiconductor questions accuracy: 0.5091024220357765
- Stable questions accuracy: 0.2723602976096248 # 반대로 해서 다시해야 함
- Oxide questions accuracy: 0.5672787715687827

Trained in GPT texts - cgcnn - galax
<!-- - Composition questions accuracy: 0.9943802437866076
- Structure questions accuracy: 0.18244419819534588  # 반대로 해서 다시 해야함 
- Metal questions accuracy: 0.3043375019787874
- Semiconductor questions accuracy: 0.4762545512110179
- Stable questions accuracy: 0.697799588412221  # 반대로 해서 다시해야 함
- Oxide questions accuracy: 0.9205318980528732 -->

Trained in merged - painn
- Composition questions accuracy: 0.9123001424726928
- Structure questions accuracy: 0.2056355865125851  # 반대로 해서 다시 해야함 
- Metal questions accuracy: 0.5908659173658383
- Semiconductor questions accuracy: 0.5721070128225424
- Stable questions accuracy: 0.6188063954408738 # 반대로 해서 다시해야 함
- Oxide questions accuracy: 0.5672787715687827


G to T Retrieval
- Trained in GPT texts
    - GPT texts
        - Batch size 64: Top1: 0.9508248567581177, Top3: 0.9964308142662048, Top10: 1.0
        - Batch size 512: Top1: 0.8682454228401184, Top3: 0.982666015625, Top10: 0.9996744990348816
        - Batch size 1024: Top1: 0.8004557490348816, Top3: 0.9544270634651184, Top10: 0.99755859375
        - Batch size 4096: Top1: 0.609619140625, Top3: 0.8310546875, Top10: 0.9602864384651184
    - academic paper paragraphs
        - Batch size 64: Top1: 0.25577694177627563, Top3: 0.5197871327400208, Top10: 0.749613344669342
        - Batch size 512: Top1: 0.12873698770999908, Top3: 0.2762643098831177, Top10: 0.4573994576931
        - Batch size 1024: Top1: 0.0978388711810112, Top3: 0.21447481215000153, Top10: 0.3704916834831238
        - Batch size 4096: Top1: 0.05019531399011612, Top3: 0.1182861328125, Top10: 0.22578124701976776
- Trained in academic paper paragraphs
    - GPT texts
        - Batch size 64: Top1: 0.7327094078063965, Top3: 0.8990323543548584, Top10: 0.9767608046531677
        - Batch size 512: Top1: 0.5131022334098816, Top3: 0.717529296875, Top10: 0.864990234375
        - Batch size 1024: Top1: 0.41943359375, Top3: 0.62548828125, Top10: 0.7950846552848816
        - Batch size 4096: Top1: 0.2482096403837204, Top3: 0.424560546875, Top10: 0.6201985478401184
    - academic paper paragraphs
        - Batch size 64: Top1: 0.3830285668373108, Top3: 0.786344587802887, Top10: 0.9453693628311157
        - Batch size 512: Top1: 0.23770707845687866, Top3: 0.531059980392456, Top10: 0.7896772623062134
        - Batch size 1024: Top1: 0.18231156468391418, Top3: 0.4258275032043457, Top10: 0.7034652233123779
        - Batch size 4096: Top1: 0.09169922024011612, Top3: 0.23024901747703552, Top10: 0.4768310487270355


T to G Retrieval
- Trained in GPT texts
    - GPT texts
        - Batch size 64: Top1: 0.9621669054031372, Top3: 0.997223973274231, Top10: 0.999920666217804
        - Batch size 512: Top1: 0.9109700322151184, Top3: 0.9892578125, Top10: 0.9994303584098816
        - Batch size 1024: Top1: 0.8617350459098816, Top3: 0.971923828125, Top10: 0.9982096552848816
        - Batch size 4096: Top1: 0.7225748896598816, Top3: 0.8949381709098816, Top10: 0.9778645634651184
    - academic paper paragraphs
        - Batch size 64: Top1: 0.24838519096374512, Top3: 0.5171715617179871, Top10: 0.7433587908744812
        - Batch size 512: Top1: 0.125594362616539, Top3: 0.2704254984855652, Top10: 0.4502909779548645
        - Batch size 1024: Top1: 0.0982198491692543, Top3: 0.21008551120758057, Top10: 0.36338451504707336
        - Batch size 4096: Top1: 0.05190429836511612, Top3: 0.11823730170726776, Top10: 0.22512206435203552
- Trained in academic paper paragraphs
    - GPT texts
        - Batch size 64: Top1: 0.7403236031532288, Top3: 0.893401026725769, Top10: 0.9665291905403137
        - Batch size 512: Top1: 0.5318196415901184, Top3: 0.7183430790901184, Top10: 0.8545735478401184
        - Batch size 1024: Top1: 0.4434407651424408, Top3: 0.6319987177848816, Top10: 0.7904459834098816
        - Batch size 4096: Top1: 0.2794596254825592, Top3: 0.4469400942325592, Top10: 0.6273600459098816
    - academic paper paragraphs
        - Batch size 64: Top1: 0.4086608588695526, Top3: 0.7873908281326294, Top10: 0.9380913376808167
        - Batch size 512: Top1: 0.2566699683666229, Top3: 0.5374814867973328, Top10: 0.77848219871521
        - Batch size 1024: Top1: 0.20727421343326569, Top3: 0.44021567702293396, Top10: 0.691920280456543
        - Batch size 4096: Top1: 0.11564941704273224, Top3: 0.2579101622104645, Top10: 0.4828857481479645

paper-cgcnn-scibert: 
paper-painn-scibert: 0.2
gpt-painn-galax: 0.15384615384615385
gpt-cgcnn-galax: 0.2
merged-cgcnn-scibert: 0.2153846153846154