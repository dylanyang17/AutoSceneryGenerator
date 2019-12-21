@echo off
if "%1" == "" (
    echo Format: test.bat ^<epoch index^> [all]
) else (
    mkdir train\epoch%1
    if "%2" == "" (
        scp -r -P 9000 student@39.104.61.196:/home/student/Work/AutoSceneryGenerator/codes/wgan_yyr/train/epoch%1/img.png ./train/epoch%1/
    ) else if "%2" == "all" (
        scp -r -P 9000 student@39.104.61.196:/home/student/Work/AutoSceneryGenerator/codes/wgan_yyr/train/epoch%1/* ./train/epoch%1/
    )
)
