@echo off
echo Starting Junon Time Series Platform...
echo Cleaning up old containers if necessary...
docker-compose down --remove-orphans

echo.
echo Building and starting containers...
docker-compose up --build -d

echo.
echo ===================================================
echo Junon Time Series Platform is RUNNING
echo ===================================================
echo Frontend: http://localhost:3000
echo Backend API: http://localhost:8000/docs
echo.
echo Logs: docker-compose logs -f
echo Stop: docker-compose down
echo ===================================================
pause
