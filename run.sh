# for mac os
scons platform=macos target=editor dev_build=yes vulkan=no

# update the app bundle with the latest binary
./update_app_bundle.sh

# run the Orca app bundle with proper dock icon
open -n ./bin/Orca.app