mt5_path = "./pretrained_weight/mt5-base"

# label paths
train_label_paths = {
                    "YT-ASL":"/data_benchmark/Uni-Sign/data/yt_asl_only.cleaned.json",
                    #"CSL_News": "./data/CSL_News/CSL_News_Labels.json",
                    "CSL_News":"/data_benchmark/Uni-Sign/data/merged_output.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.train",
                    "WLASL": "./data/WLASL/labels-2000.train"
                    }

dev_label_paths = {
                    "YT-ASL":"/data_benchmark/Uni-Sign/data/yt_asl_only.cleaned.json",
                    #"CSL_News": "./data/CSL_News/CSL_News_Labels.json",
                    "CSL_News":"/data_benchmark/Uni-Sign/data/merged_output.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.dev",
                    "WLASL": "./data/WLASL/labels-2000.dev"
                    }

test_label_paths = {
                    "YT-ASL":"/data_benchmark/Uni-Sign/data/yt_asl_only.cleaned.json",
                    #"CSL_News": "./data/CSL_News/CSL_News_Labels.json",
                    "CSL_News":"/data_benchmark/Uni-Sign/data/merged_output.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.test",
                    "WLASL": "./data/WLASL/labels-2000.test"
                    }


# video paths
rgb_dirs = {
            "YT-ASL":"./dataset/ASL/rgb_format",
            #"CSL_News": './dataset/CSL_News/rgb_format',
            "CSL_News": "./dataset/merged/rgb_format",
            "CSL_Daily": './dataset/CSL_Daily/sentence-crop',
            "WLASL": "./dataset/WLASL/rgb_format"
            }

# pose paths
pose_dirs = {
            "YT-ASL":"./dataset/ASL/pose_format",
            #"CSL_News": './dataset/CSL_News/rgb_format',
            "CSL_News": "./dataset/merged/rgb_format",
            "CSL_Daily": './dataset/CSL_Daily/pose_format',
            "WLASL": "./dataset/WLASL/pose_format"
            }