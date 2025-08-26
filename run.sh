# for mac os
scons platform=macos target=editor dev_build=yes vulkan=no

# run the Orca app bundle with proper dock icon
open -n ./bin/Orca.app