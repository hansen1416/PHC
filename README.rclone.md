Set rclone: https://rclone.org/drive/

`rclone config`

*to create a new remote, we can just follow the instruction*
Create gdrive credential for rlone: https://www.youtube.com/watch?v=Ze17oPwx6C0

# List directories in top level of your drive

`rclone lsd remote:`

# List all the files in your drive

`rclone ls remote:`

# To copy a local directory to a drive directory called backup

`rclone copy /home/source remote:backup`

# To check the total file number and size of remote folder

`rclone size gdrive:humos_phc_results`
`rclone lsf gdrive:humos_phc_results -R --files-only | wc -l`
`rclone lsf -R --files-only --fast-list gdrive:humos_phc_results | wc -l`

eg.
*Total objects: 12345*
*Total size: 6.789 GiB (7289123456 bytes)*

# delete files in drive

rclone delete gdrive:humos_phc_results/ --fast-list --progress --transfers=32 --checkers=64

# copy local folder to drive

rclone copy /home/hlz/datasets/humos_phc_results_part1 gdrive:humos_phc_results/ \
  --dry-run \
  --progress \
  --transfers=32 \
  --checkers=64 \
  --drive-chunk-size=256M \
  --fast-list


# copy remote folder to local

rclone copy   gdrive:humos_output  /home/hlz/datasets/humos_output_part1   --files-from=/home/hlz/repos/PHC/cmd/all_humos_part1.txt   --progress   --transfers=32   --checkers=64   --drive-chunk-size=256M   --fast-list