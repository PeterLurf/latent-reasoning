@echo off
title Latent Top
cd /d D:\Latent
set PYTHONIOENCODING=utf-8
python scratch\latent_top.py %*
if errorlevel 1 pause
