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

eg.
*Total objects: 12345*
*Total size: 6.789 GiB (7289123456 bytes)*