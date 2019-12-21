@echo off
if "%1" == "" (
    echo Format: test.bat ^<epoch index^>
) else (
    mkdir train\epoch%1
    scp -r -P 9000 student@39.104.61.196:/home/student/Work/AutoSceneryGenerator/codes/dcgan_yyr/train/epoch%1/img.png ./train/epoch%1/
)
