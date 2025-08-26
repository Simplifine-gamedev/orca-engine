# for mac os
scons platform=macos target=editor dev_build=yes vulkan=no

# ensure friendly binary name
ln -sf ./godot.macos.editor.dev.arm64 ./bin/Orca

# run
open -n ./bin/Orca.app