{ pkgs ? import <nixpkgs> {} }:

with pkgs;

mkShell {
  buildInputs = [
    pkgs.uv
  ];

  shellHook = ''    
    uv --version
    sh install-venv.sh
    sh activate-venv.sh
  '';
}
