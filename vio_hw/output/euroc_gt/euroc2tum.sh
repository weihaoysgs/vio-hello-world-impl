#!/bin/bash

#!/bin/bash

# 进入包含文件夹的目录
cd /path/to/your/directory

# 遍历所有文件夹
for folder in */; do
    # 忽略非文件夹的项
    [ -d "$folder" ] || continue

    # 进入文件夹并执行相应的操作
    echo "Entering folder: $folder"
    cd "$folder" || exit

    # 在这里执行你的命令，比如：
    # your_command_here
    evo_traj euroc ./data.csv --save_as_tum

    # 返回到上级目录
    cd ..
done


