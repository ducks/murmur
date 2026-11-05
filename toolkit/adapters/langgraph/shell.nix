{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  packages = [ pkgs.rustc pkgs.cargo (pkgs.python3.withPackages (ps: [ ps.langgraph ps.httpx ])) ];
}
