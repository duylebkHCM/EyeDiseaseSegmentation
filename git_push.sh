#!/bin/bash

echo "Enter your message"
read message

git add .
git config --global user.email "duy.le.bku_16039@hcmut.edu.vn"
git config --global use.name "duylebkHCM"

git commit -m"${message}"

git status
echo "Pushing data to remote server!!!"
git push -u origin main
