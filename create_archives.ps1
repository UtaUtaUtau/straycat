Write-Host "Creating archives..."
Copy-Item -Path .\straycat_server.py -Destination .\publish\
Copy-Item -Path .\straycat_server.py -Destination .\publish\mac\
Copy-Item -Path .\straycat_server.py -Destination .\publish\linux\
Write-Host "Creating Windows archive..."
Compress-Archive -Path .\publish\* -DestinationPath .\publish\StrayCatRunner_Windows.zip
Write-Host "Creating Mac archive..."
Compress-Archive -Path .\publish\mac\* -DestinationPath .\publish\StrayCatRunner_Mac.zip
Write-Host "Creating Linux archive..."
Compress-Archive -Path .\publish\linux\* -DestinationPath .\publish\StrayCatRunner_Linux.zip