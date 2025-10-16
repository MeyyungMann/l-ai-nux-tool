# Online Mode - 50 Query Test Report

**Generated**: 2025-10-14 18:59:44  
**Mode**: Online (API-based)  
**Model**: gpt-5-mini  
**Duration**: 253.7 seconds  
**Success Rate**: 98.0% (49/50)

---

## 📊 Summary Statistics

### Overall Results
- ✅ **Successful**: 49 queries
- ❌ **Failed**: 1 queries
- ⏱️ **Avg Time**: 5.07s per query

### By Category

| Category | Total | Success | Failed | Success Rate |
|----------|-------|---------|--------|--------------|
| Directory | 8 | 8 | 0 | 100.0% |
| File Ops | 10 | 10 | 0 | 100.0% |
| Process | 7 | 7 | 0 | 100.0% |
| Search | 10 | 9 | 1 | 90.0% |
| System | 7 | 7 | 0 | 100.0% |
| Text | 8 | 8 | 0 | 100.0% |

---

## ✅ Successful Commands

### 1. list all files including hidden ones
- **Category**: File Ops
- **Command**: `ls -la`
- **RAG Recommendation**: ✅ Frequently used command - would improve offline mode

### 2. create a new file named test.txt
- **Category**: File Ops
- **Command**: `touch test.txt`
- **RAG Recommendation**: ✅ Frequently used command - would improve offline mode

### 3. copy file1.txt to file2.txt
- **Category**: File Ops
- **Command**: `cp file1.txt file2.txt`
- **RAG Recommendation**: ✅ Frequently used command - would improve offline mode

### 4. move old.log to archive folder
- **Category**: File Ops
- **Command**: `mv old.log archive/`
- **RAG Recommendation**: ✅ Frequently used command - would improve offline mode

### 5. delete all txt files in current directory
- **Category**: File Ops
- **Command**: `find . -maxdepth 1 -type f -name "*.txt" -exec rm -f -- ls -la +`
- **RAG Recommendation**: ✅ Frequently used command - would improve offline mode

### 6. rename document.pdf to report.pdf
- **Category**: File Ops
- **Command**: `mv document.pdf report.pdf`
- **RAG Recommendation**: ✅ Frequently used command - would improve offline mode

### 7. create backup of config.json
- **Category**: File Ops
- **Command**: `cp config.json config.json.bak`
- **RAG Recommendation**: ✅ Frequently used command - would improve offline mode

### 8. show file permissions of script.sh
- **Category**: File Ops
- **Command**: `ls -l script.sh`
- **RAG Recommendation**: ✅ Frequently used command - would improve offline mode

### 9. make script.sh executable
- **Category**: File Ops
- **Command**: `chmod +x script.sh`
- **RAG Recommendation**: ✅ Frequently used command - would improve offline mode

### 10. change owner of file.txt to root
- **Category**: File Ops
- **Command**: `sudo chown root file.txt`
- **RAG Recommendation**: ✅ Frequently used command - would improve offline mode

### 11. show current working directory
- **Category**: Directory
- **Command**: `pwd`
- **RAG Recommendation**: ✅ Frequently used command - would improve offline mode

### 12. create a new directory called projects
- **Category**: Directory
- **Command**: `mkdir -p projects`
- **RAG Recommendation**: ✅ Frequently used command - would improve offline mode

### 13. list directory tree structure
- **Category**: Directory
- **Command**: `tree -a`
- **RAG Recommendation**: ✅ Frequently used command - would improve offline mode

### 14. remove empty directory temp
- **Category**: Directory
- **Command**: `rmdir temp`
- **RAG Recommendation**: ✅ Frequently used command - would improve offline mode

### 15. create nested directories a/b/c
- **Category**: Directory
- **Command**: `mkdir -p a/b/c`
- **RAG Recommendation**: ✅ Frequently used command - would improve offline mode

### 16. change to home directory
- **Category**: Directory
- **Command**: `cd ~`
- **RAG Recommendation**: ✅ Frequently used command - would improve offline mode

### 17. count files in current directory
- **Category**: Directory
- **Command**: `find . -maxdepth 1 -type f | wc -l`
- **RAG Recommendation**: ✅ Frequently used command - would improve offline mode

### 18. show disk usage of current directory
- **Category**: Directory
- **Command**: `du -sh .`
- **RAG Recommendation**: ✅ Frequently used command - would improve offline mode

### 19. find all jpg files recursively
- **Category**: Search
- **Command**: `find . -type f -iname "*.jpg"`
- **RAG Recommendation**: ✅ Frequently used command - would improve offline mode

### 20. find files modified in last 7 days
- **Category**: Search
- **Command**: `find . -type f -mtime -7`
- **RAG Recommendation**: ✅ Frequently used command - would improve offline mode

### 21. find files larger than 100MB
- **Category**: Search
- **Command**: `find . -type f -size +100M`
- **RAG Recommendation**: ✅ Frequently used command - would improve offline mode

### 22. find empty files
- **Category**: Search
- **Command**: `find . -type f -empty`
- **RAG Recommendation**: ✅ Frequently used command - would improve offline mode

### 23. search for text 'error' in all log files
- **Category**: Search
- **Command**: `grep -RIn --include="*.log" "error" .`
- **RAG Recommendation**: ✅ Frequently used command - would improve offline mode

### 24. find files owned by current user
- **Category**: Search
- **Command**: `find . -user "$(whoami)"`
- **RAG Recommendation**: ✅ Frequently used command - would improve offline mode

### 25. locate python executable
- **Category**: Search
- **Command**: `command -v python || command -v python3 || type -a python python3`
- **RAG Recommendation**: ✅ Frequently used command - would improve offline mode

### 27. search for pattern in all sh files
- **Category**: Search
- **Command**: `read -p "Enter pattern: " p && grep -nR --include="*.sh" -- "$p" .`
- **RAG Recommendation**: ✅ Frequently used command - would improve offline mode

### 28. find files with 777 permissions
- **Category**: Search
- **Command**: `find . -type f -perm 0777`
- **RAG Recommendation**: ✅ Frequently used command - would improve offline mode

### 29. show first 10 lines of log.txt
- **Category**: Text
- **Command**: `head -n 10 log.txt`
- **RAG Recommendation**: ✅ Frequently used command - would improve offline mode

### 30. show last 20 lines of error.log
- **Category**: Text
- **Command**: `tail -n 20 error.log`
- **RAG Recommendation**: ✅ Frequently used command - would improve offline mode

### 31. count lines in file.txt
- **Category**: Text
- **Command**: `wc -l file.txt`
- **RAG Recommendation**: ✅ Frequently used command - would improve offline mode

### 32. count words in document.txt
- **Category**: Text
- **Command**: `wc -w document.txt`
- **RAG Recommendation**: ✅ Frequently used command - would improve offline mode

### 33. sort lines in names.txt alphabetically
- **Category**: Text
- **Command**: `sort -o names.txt names.txt`
- **RAG Recommendation**: ✅ Frequently used command - would improve offline mode

### 34. remove duplicate lines from data.txt
- **Category**: Text
- **Command**: `tmp=$(mktemp) && awk '!seen[$0]++' data.txt > "$tmp" && mv "$tmp" data.txt`
- **RAG Recommendation**: ✅ Frequently used command - would improve offline mode

### 35. replace 'old' with 'new' in file.txt
- **Category**: Text
- **Command**: `sed -i 's/old/new/g' file.txt`
- **RAG Recommendation**: ✅ Frequently used command - would improve offline mode

### 36. extract column 2 from csv.txt
- **Category**: Text
- **Command**: `cut -d',' -f2 csv.txt`
- **RAG Recommendation**: ✅ Frequently used command - would improve offline mode

### 37. show system information
- **Category**: System
- **Command**: `uname -a`
- **RAG Recommendation**: ✅ Useful system command - good RAG addition

### 38. display current date and time
- **Category**: System
- **Command**: `date`
- **RAG Recommendation**: ✅ Useful system command - good RAG addition

### 39. show memory usage
- **Category**: System
- **Command**: `free -h`
- **RAG Recommendation**: ✅ Useful system command - good RAG addition

### 40. show cpu information
- **Category**: System
- **Command**: `lscpu`
- **RAG Recommendation**: ✅ Useful system command - good RAG addition

### 41. check disk space
- **Category**: System
- **Command**: `df -h`
- **RAG Recommendation**: ✅ Useful system command - good RAG addition

### 42. show logged in users
- **Category**: System
- **Command**: `who`
- **RAG Recommendation**: ✅ Useful system command - good RAG addition

### 43. display system uptime
- **Category**: System
- **Command**: `uptime`
- **RAG Recommendation**: ✅ Useful system command - good RAG addition

### 44. list all running processes
- **Category**: Process
- **Command**: `ps aux`
- **RAG Recommendation**: ✅ Useful system command - good RAG addition

### 45. show top 10 cpu consuming processes
- **Category**: Process
- **Command**: `ps aux --sort=-%cpu | head -n 11`
- **RAG Recommendation**: ✅ Useful system command - good RAG addition

### 46. find process by name nginx
- **Category**: Process
- **Command**: `pgrep -a nginx`
- **RAG Recommendation**: ✅ Useful system command - good RAG addition

### 47. kill process with pid 1234
- **Category**: Process
- **Command**: `kill 1234`
- **RAG Recommendation**: ✅ Useful system command - good RAG addition

### 48. show processes using port 8080
- **Category**: Process
- **Command**: `lsof -nP -i :8080`
- **RAG Recommendation**: ✅ Useful system command - good RAG addition

### 49. monitor system resources in real-time
- **Category**: Process
- **Command**: `top`
- **RAG Recommendation**: ✅ Useful system command - good RAG addition

### 50. show memory usage per process
- **Category**: Process
- **Command**: `ps aux --sort=-%mem`
- **RAG Recommendation**: ✅ Useful system command - good RAG addition

---

## ❌ Failed Commands

### 26. find duplicate files by name
- **Category**: Search
- **Generated**: ``
- **Failure Type**: EMPTY_OUTPUT
- **Reason**: Model generated empty response
- **Error**: Invalid command structure: 

---

## 🔍 Failure Analysis

### Failure Types Breakdown

#### EMPTY_OUTPUT (1 occurrences)

- **Query**: find duplicate files by name
  - Generated: ``
  - Reason: Model generated empty response

---

## 💾 RAG Cache Recommendations

These commands should be added to RAG cache to improve offline mode performance:

**Total Recommendations**: 49 commands

### Directory (8 commands)

#### show current working directory
```bash
pwd
```
*Reason*: Frequently used command - would improve offline mode

#### create a new directory called projects
```bash
mkdir -p projects
```
*Reason*: Frequently used command - would improve offline mode

#### list directory tree structure
```bash
tree -a
```
*Reason*: Frequently used command - would improve offline mode

#### remove empty directory temp
```bash
rmdir temp
```
*Reason*: Frequently used command - would improve offline mode

#### create nested directories a/b/c
```bash
mkdir -p a/b/c
```
*Reason*: Frequently used command - would improve offline mode

#### change to home directory
```bash
cd ~
```
*Reason*: Frequently used command - would improve offline mode

#### count files in current directory
```bash
find . -maxdepth 1 -type f | wc -l
```
*Reason*: Frequently used command - would improve offline mode

#### show disk usage of current directory
```bash
du -sh .
```
*Reason*: Frequently used command - would improve offline mode

### File Ops (10 commands)

#### list all files including hidden ones
```bash
ls -la
```
*Reason*: Frequently used command - would improve offline mode

#### create a new file named test.txt
```bash
touch test.txt
```
*Reason*: Frequently used command - would improve offline mode

#### copy file1.txt to file2.txt
```bash
cp file1.txt file2.txt
```
*Reason*: Frequently used command - would improve offline mode

#### move old.log to archive folder
```bash
mv old.log archive/
```
*Reason*: Frequently used command - would improve offline mode

#### delete all txt files in current directory
```bash
find . -maxdepth 1 -type f -name "*.txt" -exec rm -f -- ls -la +
```
*Reason*: Frequently used command - would improve offline mode

#### rename document.pdf to report.pdf
```bash
mv document.pdf report.pdf
```
*Reason*: Frequently used command - would improve offline mode

#### create backup of config.json
```bash
cp config.json config.json.bak
```
*Reason*: Frequently used command - would improve offline mode

#### show file permissions of script.sh
```bash
ls -l script.sh
```
*Reason*: Frequently used command - would improve offline mode

#### make script.sh executable
```bash
chmod +x script.sh
```
*Reason*: Frequently used command - would improve offline mode

#### change owner of file.txt to root
```bash
sudo chown root file.txt
```
*Reason*: Frequently used command - would improve offline mode

### Process (7 commands)

#### list all running processes
```bash
ps aux
```
*Reason*: Useful system command - good RAG addition

#### show top 10 cpu consuming processes
```bash
ps aux --sort=-%cpu | head -n 11
```
*Reason*: Useful system command - good RAG addition

#### find process by name nginx
```bash
pgrep -a nginx
```
*Reason*: Useful system command - good RAG addition

#### kill process with pid 1234
```bash
kill 1234
```
*Reason*: Useful system command - good RAG addition

#### show processes using port 8080
```bash
lsof -nP -i :8080
```
*Reason*: Useful system command - good RAG addition

#### monitor system resources in real-time
```bash
top
```
*Reason*: Useful system command - good RAG addition

#### show memory usage per process
```bash
ps aux --sort=-%mem
```
*Reason*: Useful system command - good RAG addition

### Search (9 commands)

#### find all jpg files recursively
```bash
find . -type f -iname "*.jpg"
```
*Reason*: Frequently used command - would improve offline mode

#### find files modified in last 7 days
```bash
find . -type f -mtime -7
```
*Reason*: Frequently used command - would improve offline mode

#### find files larger than 100MB
```bash
find . -type f -size +100M
```
*Reason*: Frequently used command - would improve offline mode

#### find empty files
```bash
find . -type f -empty
```
*Reason*: Frequently used command - would improve offline mode

#### search for text 'error' in all log files
```bash
grep -RIn --include="*.log" "error" .
```
*Reason*: Frequently used command - would improve offline mode

#### find files owned by current user
```bash
find . -user "$(whoami)"
```
*Reason*: Frequently used command - would improve offline mode

#### locate python executable
```bash
command -v python || command -v python3 || type -a python python3
```
*Reason*: Frequently used command - would improve offline mode

#### search for pattern in all sh files
```bash
read -p "Enter pattern: " p && grep -nR --include="*.sh" -- "$p" .
```
*Reason*: Frequently used command - would improve offline mode

#### find files with 777 permissions
```bash
find . -type f -perm 0777
```
*Reason*: Frequently used command - would improve offline mode

### System (7 commands)

#### show system information
```bash
uname -a
```
*Reason*: Useful system command - good RAG addition

#### display current date and time
```bash
date
```
*Reason*: Useful system command - good RAG addition

#### show memory usage
```bash
free -h
```
*Reason*: Useful system command - good RAG addition

#### show cpu information
```bash
lscpu
```
*Reason*: Useful system command - good RAG addition

#### check disk space
```bash
df -h
```
*Reason*: Useful system command - good RAG addition

#### show logged in users
```bash
who
```
*Reason*: Useful system command - good RAG addition

#### display system uptime
```bash
uptime
```
*Reason*: Useful system command - good RAG addition

### Text (8 commands)

#### show first 10 lines of log.txt
```bash
head -n 10 log.txt
```
*Reason*: Frequently used command - would improve offline mode

#### show last 20 lines of error.log
```bash
tail -n 20 error.log
```
*Reason*: Frequently used command - would improve offline mode

#### count lines in file.txt
```bash
wc -l file.txt
```
*Reason*: Frequently used command - would improve offline mode

#### count words in document.txt
```bash
wc -w document.txt
```
*Reason*: Frequently used command - would improve offline mode

#### sort lines in names.txt alphabetically
```bash
sort -o names.txt names.txt
```
*Reason*: Frequently used command - would improve offline mode

#### remove duplicate lines from data.txt
```bash
tmp=$(mktemp) && awk '!seen[$0]++' data.txt > "$tmp" && mv "$tmp" data.txt
```
*Reason*: Frequently used command - would improve offline mode

#### replace 'old' with 'new' in file.txt
```bash
sed -i 's/old/new/g' file.txt
```
*Reason*: Frequently used command - would improve offline mode

#### extract column 2 from csv.txt
```bash
cut -d',' -f2 csv.txt
```
*Reason*: Frequently used command - would improve offline mode

---

## 🎯 Quality Analysis

### Command Quality Metrics

- **Average Command Length**: 20.2 characters
- **Commands with Flags**: 31/50 (62.0%)
- **Commands with Quotes**: 8/50 (16.0%)

### Command Complexity Distribution

- **Simple** (1-2 words): 16 commands
- **Medium** (3-5 words): 23 commands
- **Complex** (6+ words): 10 commands

---

## 📋 Complete Test Results

| # | Category | Query | Command | Status |
|---|----------|-------|---------|--------|
| 1 | File Ops | list all files including hidden ones... | `ls -la` | ✅ |
| 2 | File Ops | create a new file named test.txt... | `touch test.txt` | ✅ |
| 3 | File Ops | copy file1.txt to file2.txt... | `cp file1.txt file2.txt` | ✅ |
| 4 | File Ops | move old.log to archive folder... | `mv old.log archive/` | ✅ |
| 5 | File Ops | delete all txt files in current director... | `find . -maxdepth 1 -type f -name "*.txt" -exec rm ...` | ✅ |
| 6 | File Ops | rename document.pdf to report.pdf... | `mv document.pdf report.pdf` | ✅ |
| 7 | File Ops | create backup of config.json... | `cp config.json config.json.bak` | ✅ |
| 8 | File Ops | show file permissions of script.sh... | `ls -l script.sh` | ✅ |
| 9 | File Ops | make script.sh executable... | `chmod +x script.sh` | ✅ |
| 10 | File Ops | change owner of file.txt to root... | `sudo chown root file.txt` | ✅ |
| 11 | Directory | show current working directory... | `pwd` | ✅ |
| 12 | Directory | create a new directory called projects... | `mkdir -p projects` | ✅ |
| 13 | Directory | list directory tree structure... | `tree -a` | ✅ |
| 14 | Directory | remove empty directory temp... | `rmdir temp` | ✅ |
| 15 | Directory | create nested directories a/b/c... | `mkdir -p a/b/c` | ✅ |
| 16 | Directory | change to home directory... | `cd ~` | ✅ |
| 17 | Directory | count files in current directory... | `find . -maxdepth 1 -type f | wc -l` | ✅ |
| 18 | Directory | show disk usage of current directory... | `du -sh .` | ✅ |
| 19 | Search | find all jpg files recursively... | `find . -type f -iname "*.jpg"` | ✅ |
| 20 | Search | find files modified in last 7 days... | `find . -type f -mtime -7` | ✅ |
| 21 | Search | find files larger than 100MB... | `find . -type f -size +100M` | ✅ |
| 22 | Search | find empty files... | `find . -type f -empty` | ✅ |
| 23 | Search | search for text 'error' in all log files... | `grep -RIn --include="*.log" "error" .` | ✅ |
| 24 | Search | find files owned by current user... | `find . -user "$(whoami)"` | ✅ |
| 25 | Search | locate python executable... | `command -v python || command -v python3 || type -a...` | ✅ |
| 26 | Search | find duplicate files by name... | `` | ❌ |
| 27 | Search | search for pattern in all sh files... | `read -p "Enter pattern: " p && grep -nR --include=...` | ✅ |
| 28 | Search | find files with 777 permissions... | `find . -type f -perm 0777` | ✅ |
| 29 | Text | show first 10 lines of log.txt... | `head -n 10 log.txt` | ✅ |
| 30 | Text | show last 20 lines of error.log... | `tail -n 20 error.log` | ✅ |
| 31 | Text | count lines in file.txt... | `wc -l file.txt` | ✅ |
| 32 | Text | count words in document.txt... | `wc -w document.txt` | ✅ |
| 33 | Text | sort lines in names.txt alphabetically... | `sort -o names.txt names.txt` | ✅ |
| 34 | Text | remove duplicate lines from data.txt... | `tmp=$(mktemp) && awk '!seen[$0]++' data.txt > "$tm...` | ✅ |
| 35 | Text | replace 'old' with 'new' in file.txt... | `sed -i 's/old/new/g' file.txt` | ✅ |
| 36 | Text | extract column 2 from csv.txt... | `cut -d',' -f2 csv.txt` | ✅ |
| 37 | System | show system information... | `uname -a` | ✅ |
| 38 | System | display current date and time... | `date` | ✅ |
| 39 | System | show memory usage... | `free -h` | ✅ |
| 40 | System | show cpu information... | `lscpu` | ✅ |
| 41 | System | check disk space... | `df -h` | ✅ |
| 42 | System | show logged in users... | `who` | ✅ |
| 43 | System | display system uptime... | `uptime` | ✅ |
| 44 | Process | list all running processes... | `ps aux` | ✅ |
| 45 | Process | show top 10 cpu consuming processes... | `ps aux --sort=-%cpu | head -n 11` | ✅ |
| 46 | Process | find process by name nginx... | `pgrep -a nginx` | ✅ |
| 47 | Process | kill process with pid 1234... | `kill 1234` | ✅ |
| 48 | Process | show processes using port 8080... | `lsof -nP -i :8080` | ✅ |
| 49 | Process | monitor system resources in real-time... | `top` | ✅ |
| 50 | Process | show memory usage per process... | `ps aux --sort=-%mem` | ✅ |


---

## 🚀 Recommendations

### High Priority RAG Additions (35 commands)

These commands are frequently used and should be added to RAG immediately:

- `ls -la` - list all files including hidden ones
- `touch test.txt` - create a new file named test.txt
- `cp file1.txt file2.txt` - copy file1.txt to file2.txt
- `mv old.log archive/` - move old.log to archive folder
- `find . -maxdepth 1 -type f -name "*.txt" -exec rm -f -- ls -la +` - delete all txt files in current directory
- `mv document.pdf report.pdf` - rename document.pdf to report.pdf
- `cp config.json config.json.bak` - create backup of config.json
- `ls -l script.sh` - show file permissions of script.sh
- `chmod +x script.sh` - make script.sh executable
- `sudo chown root file.txt` - change owner of file.txt to root

### Medium Priority RAG Additions (14 commands)

Useful but less common:

- `uname -a` - show system information
- `date` - display current date and time
- `free -h` - show memory usage
- `lscpu` - show cpu information
- `df -h` - check disk space
- `who` - show logged in users
- `uptime` - display system uptime
- `ps aux` - list all running processes
- `ps aux --sort=-%cpu | head -n 11` - show top 10 cpu consuming processes
- `pgrep -a nginx` - find process by name nginx


### Failed Commands Requiring Attention (1 commands)

These queries failed and need investigation:

- **find duplicate files by name**: EMPTY_OUTPUT - Model generated empty response


---

## 💡 Insights & Next Steps

### What Worked Well
- ✅ Very high success rate - online mode is highly reliable
- ✅ Generated valid commands for 49 different tasks
- ✅ Covered 6 different command categories

### Areas for Improvement
- ⚠️  1 different failure types detected
  - EMPTY_OUTPUT: 1 occurrences

### RAG Cache Strategy

1. **Immediate Additions** (35 commands):
   - Add high-priority frequently-used commands to RAG
   - Will significantly improve offline mode accuracy
   
2. **Gradual Additions** (14 commands):
   - Add medium-priority commands as users request them
   - Use execution fallback to learn from successful online commands
   
3. **Monitor & Iterate**:
   - Track which commands users request most
   - Automatically add successful execution-fallback commands
   - Continuously improve offline mode

---

## 🔧 Implementation Commands

### Add Recommended Commands to RAG

```python
from src.rag_engine import get_rag_engine
from pathlib import Path
import json

rag = get_rag_engine()

# High-priority commands
recommendations = [
    ("list all files including hidden ones", "ls -la"),
    ("create a new file named test.txt", "touch test.txt"),
    ("copy file1.txt to file2.txt", "cp file1.txt file2.txt"),
    ("move old.log to archive folder", "mv old.log archive/"),
    ("delete all txt files in current directory", "find . -maxdepth 1 -type f -name "*.txt" -exec rm -f -- ls -la +"),
]

for query, command in recommendations:
    context = f"Known-good command for this task:\n\n{command}"
    rag.store_context_for_query(query, context, top_k=3)
    print(f"Added: {query} → {command}")
```

### Export to user_approved.jsonl

```bash
# These commands can be automatically promoted to RAG on next startup
cat >> ~/.lai-nux-tool/rag_cache/user_approved.jsonl << EOF
{"description": "list all files including hidden ones", "command": "ls -la"}
{"description": "create a new file named test.txt", "command": "touch test.txt"}
{"description": "copy file1.txt to file2.txt", "command": "cp file1.txt file2.txt"}
{"description": "move old.log to archive folder", "command": "mv old.log archive/"}
{"description": "delete all txt files in current directory", "command": "find . -maxdepth 1 -type f -name "*.txt" -exec rm -f -- ls -la +"}
EOF
```

---

## 📈 Historical Comparison

*Track improvements over time by running this test periodically*

| Date | Success Rate | Avg Time | RAG Additions |
|------|--------------|----------|---------------|
| 2025-10-14 | 98.0% | 5.07s | 49 recommended |

---

**Test Status**: ✅ Complete  
**Report Version**: 1.0  
**Next Test**: Run after adding RAG recommendations
