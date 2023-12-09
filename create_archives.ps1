Copy-Item -Path .\straycat_server.py -Destination .\publish\
Copy-Item -Path .\straycat_server.py -Destination .\publish\mac\
Copy-Item -Path .\straycat_server.py -Destination .\publish\linux\

Compress-Archive -Path .\publish\* -DestinationPath .\publish\StrayCatRunner_Windows.zip
Compress-Archive -Path .\publish\mac\* -DestinationPath .\publish\StrayCatRunner_Mac.zip
Compress-Archive -Path .\publish\linux\* -DestinationPath .\publish\StrayCatRunner_Linux.zip