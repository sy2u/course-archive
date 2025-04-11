# 📚 Course Archive

This repository collects projects from my undergraduate studies in Electrical and Computer Engineering at **UIUC** and **ZJU**. Topics span systems programming, computer architecture, data structures, applied parallel programming, VLSI design, and more.

## 🗂 Structure

Each folder corresponds to a specific course, following the naming convention: CourseCode_ShortName

| Folder                | Course                                |
|----------------------|----------------------------------------|
| `CS225_DataStructs`  | Data Structures                        |
| `ECE220_SysProg`     | Computer Systems & Programming         |
| `ECE391_CompSys`     | Computer Systems Engineering           |
| `ECE408_ParProg`     | Applied Parallel Programming           |
| `ECE411_CompOrg`     | Computer Organization and Design       |
| `ECE425_VLSI`        | Intro to VLSI System Design            |

## 🔗 Submodules

Some projects are included as Git submodules. For example:

- `ECE411_CompOrg/mp_OoO` – an Out-of-Order RISC-V Processor

To clone this repository along with all submodules:

```bash
git clone --recurse-submodules git@github.com:sy2u/course-archive.git
```
Or, if you've already cloned it:
```bash
git submodule update --init --recursive
```
