package = "csream"
version = "scm-1"

source = {
   url = "git://github.com/ludc/csream.git",
}

description = {
   summary = "Cost-Sensitive Sequential Models",
   detailed = [[
   ]],
   homepage = "https://github.com/ludc/csream",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
]],
   install_command = "cd build && echo $(PREFIX) && $(MAKE) install"
}
