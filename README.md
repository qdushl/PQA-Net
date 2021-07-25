# PQA-Net
MainDTLQ.py is for distortion classification task and please run it first.
The explanations of some parameters are as below: 
 
    parser.add_argument("--trainsetDT", type=str, default="/home/qi/QiLiu/code/MEONCode/MEONLQ/dataset/")  ######## distortion tested images patch ##############
    parser.add_argument("--train_csv_DT", type=str,
                            default="/home/qi/QiLiu/code/MEONCode/MEONLQModelChange/label/PCMeon2DelDMOSSameTrainbcmp_dist.txt") ######## distortion tested images dist file  ##############
    parser.add_argument("--trainset", type=str, default="/home/qi/QiLiu/code/MEONCode/MEONLQ/dataset/") ######## quality tested images patch ##############
    parser.add_argument("--train_csv", type=str,
                        default="/home/qi/QiLiu/code/MEONCode/MEONLQModelChange/label/PCMeon2DelDMOSSameTrainbcmp_mos.txt") ######## quality tested images mos file  ##############
    parser.add_argument("--output_channel", type=int, default=4)  ######## distortion classification number  ##############
    parser.add_argument('--ckpt_path', default='./checkpoint_pretrain4DTbgpsMC_dropout/', type=str, metavar='PATH',help='path to checkpoints') ######## distortion classification model patch  ##############
    parser.add_argument('--ckpt', default="MeonDT-00016.pt", type=str, help='name of the checkpoint to load') ######## distortion classification model name  ##############
    parser.add_argument('--board', default="./board_pretrainDT6Sep_lr3", type=str, help='tensorboardX log file path')  ######## distortion classification task tensorboard path ##############

MainLQ.py is final task and please run it after finishing MainDTLQ.py.
The explanations of some parameters are as below: 
    parser.add_argument("--trainsetDT", type=str, default="/home/qi/QiLiu/code/MEONCode/MEONLQ/dataset/") ######## distortion tested images patch ##############
    parser.add_argument("--train_csv_DT", type=str,
                            default="/home/qi/QiLiu/code/MEONCode/MEONLQModelChange/label/PCMeon2DelDMOSSameTrainbcmp_dist.txt") ######## distortion tested images dist file  ##############
    parser.add_argument("--trainset", type=str, default="/home/qi/QiLiu/code/MEONCode/MEONLQ/dataset/")  ######## quality tested images patch ##############
    parser.add_argument("--train_csv", type=str,
                        default="/home/qi/QiLiu/code/MEONCode/MEONLQModelChange/label/PCMeon2DelDMOSSameTrainbcmp_mos.txt") ######## quality tested images mos file  ##############
    parser.add_argument("--output_channel", type=int, default=4)  ######## distortion classification number  ##############
    parser.add_argument('--ckpt_path', default='./checkpoint_pretrian4bgpsMC_dropout_lambda10_0.5/', type=str, metavar='PATH', help='path to checkpoints') ######## prediction quality model patch  ##############
    parser.add_argument('--ckpt', default="Meon-00023.pt", type=str, help='name of the checkpoint to load') ######## prediction quality model name  ##############
    parser.add_argument('--board', default="./board_pretrain4bgps", type=str, help='tensorboardX log file path') ######## prediction quality task tensorboard path ##############
    parser.add_argument('--DT_ckpt_path', default='./checkpoint_pretrain4DTbgpsMC_dropout/', type=str, metavar='PATH', help='path to DT checkpoints') ######## distortion classification model patch  ##############
    parser.add_argument('--DT_ckpt', default="MeonDT-00026.pt", type=str, help='name of the DT checkpoint to load') ######## distortion classification model name  ##############
    
    If you use this code please cite the paper "PQA-Net: Deep No Reference Point Cloud Quality Assessment via Multi-view Projection"
    
