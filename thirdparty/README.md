# thirdparty

该目录用于管理第三方依赖。

## CUTLASS（通过 git submodule）

本项目将 CUTLASS 配置为 submodule：`thirdparty/cutlass`。

首次拉取（或新环境）请执行：

```bash
git submodule update --init --recursive
```

若需要更新 CUTLASS 到新版本：

```bash
git submodule update --remote --recursive thirdparty/cutlass
```
