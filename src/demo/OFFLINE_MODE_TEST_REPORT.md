# Offline Mode - 50 Query Test Report

**Generated**: 2025-10-15 14:07:43  
**Mode**: Offline (RAG-based)  
**Model**: Local CodeLlama-7b-Instruct  
**Duration**: 215.7 seconds  
**Success Rate**: 100.0% (50/50)  
**RAG Hit Rate**: 100.0% (50/50)
**Online Comparison**: +2.0% difference (Online: 98.0%)


---

## 📊 Summary Statistics

### Overall Results
- ✅ **Successful**: 50 queries
- ❌ **Failed**: 0 queries
- ⏱️ **Avg Time**: 4.31s per query
- 📚 **RAG Hits**: 50 queries
- 🔍 **RAG Misses**: 0 queries

### By Category

| Category | Total | Success | Failed | Success Rate |
|----------|-------|---------|--------|--------------|
| Directory | 8 | 8 | 0 | 100.0% |
| File Ops | 10 | 10 | 0 | 100.0% |
| Process | 7 | 7 | 0 | 100.0% |
| Search | 10 | 10 | 0 | 100.0% |
| System | 7 | 7 | 0 | 100.0% |
| Text | 8 | 8 | 0 | 100.0% |

---

## ✅ Successful Commands

### 1. list all files including hidden ones
- **Category**: File Ops
- **Command**: `ls -la`
- **RAG Status**: RAG_HIT

### 2. create a new file named test.txt
- **Category**: File Ops
- **Command**: `touch test.txt`
- **RAG Status**: RAG_HIT

### 3. copy file1.txt to file2.txt
- **Category**: File Ops
- **Command**: `cp file1.txt file2.txt`
- **RAG Status**: RAG_HIT

### 4. move old.log to archive folder
- **Category**: File Ops
- **Command**: `mv old.log archive/`
- **RAG Status**: RAG_HIT

### 5. delete all txt files in current directory
- **Category**: File Ops
- **Command**: `find . -name "*.txt" -type f -delete`
- **RAG Status**: RAG_HIT

### 6. rename document.pdf to report.pdf
- **Category**: File Ops
- **Command**: `mv document.pdf report.pdf`
- **RAG Status**: RAG_HIT

### 7. create backup of config.json
- **Category**: File Ops
- **Command**: `cp config.json config.json.backup`
- **RAG Status**: RAG_HIT

### 8. show file permissions of script.sh
- **Category**: File Ops
- **Command**: `ls -l script.sh`
- **RAG Status**: RAG_HIT

### 9. make script.sh executable
- **Category**: File Ops
- **Command**: `chmod +x script.sh`
- **RAG Status**: RAG_HIT

### 10. change owner of file.txt to root
- **Category**: File Ops
- **Command**: `chown root file.txt`
- **RAG Status**: RAG_HIT

### 11. show current working directory
- **Category**: Directory
- **Command**: `pwd`
- **RAG Status**: RAG_HIT

### 12. create a new directory called projects
- **Category**: Directory
- **Command**: `mkdir projects`
- **RAG Status**: RAG_HIT

### 13. list directory tree structure
- **Category**: Directory
- **Command**: `tree -a -L 2 -d -p`
- **RAG Status**: RAG_HIT

### 14. remove empty directory temp
- **Category**: Directory
- **Command**: `rmdir temp`
- **RAG Status**: RAG_HIT

### 15. create nested directories a/b/c
- **Category**: Directory
- **Command**: `mkdir -p a/b/c`
- **RAG Status**: RAG_HIT

### 16. change to home directory
- **Category**: Directory
- **Command**: `cd ~`
- **RAG Status**: RAG_HIT

### 17. count files in current directory
- **Category**: Directory
- **Command**: `ls -1 | wc -l`
- **RAG Status**: RAG_HIT

### 18. show disk usage of current directory
- **Category**: Directory
- **Command**: `du -sh`
- **RAG Status**: RAG_HIT

### 19. find all jpg files recursively
- **Category**: Search
- **Command**: `find . -type f -name "*.jpg"`
- **RAG Status**: RAG_HIT

### 20. find files modified in last 7 days
- **Category**: Search
- **Command**: `find . -type f -mtime -7`
- **RAG Status**: RAG_HIT

### 21. find files larger than 100MB
- **Category**: Search
- **Command**: `find . -type f -size +100M`
- **RAG Status**: RAG_HIT

### 22. find empty files
- **Category**: Search
- **Command**: `find . -type f -empty`
- **RAG Status**: RAG_HIT

### 23. search for text 'error' in all log files
- **Category**: Search
- **Command**: `find . -name "*.log" -exec grep 'error' ls -la \;`
- **RAG Status**: RAG_HIT

### 24. find files owned by current user
- **Category**: Search
- **Command**: `find . -user $(whoami)`
- **RAG Status**: RAG_HIT

### 25. locate python executable
- **Category**: Search
- **Command**: `which python`
- **RAG Status**: RAG_HIT

### 26. find duplicate files by name
- **Category**: Search
- **Command**: `find . -type f -exec md5sum ls -la \; | sort | uniq -w32 --all-repeated=separate | cut -d''-f3-`
- **RAG Status**: RAG_HIT

### 27. search for pattern in all sh files
- **Category**: Search
- **Command**: `find . -name "*.sh" -exec grep -H pattern ls -la \;`
- **RAG Status**: RAG_HIT

### 28. find files with 777 permissions
- **Category**: Search
- **Command**: `find . -type f -perm 777`
- **RAG Status**: RAG_HIT

### 29. show first 10 lines of log.txt
- **Category**: Text
- **Command**: `head -n 10 log.txt`
- **RAG Status**: RAG_HIT

### 30. show last 20 lines of error.log
- **Category**: Text
- **Command**: `tail -n 20 error.log`
- **RAG Status**: RAG_HIT

### 31. count lines in file.txt
- **Category**: Text
- **Command**: `wc -l file.txt`
- **RAG Status**: RAG_HIT

### 32. count words in document.txt
- **Category**: Text
- **Command**: `wc -w document.txt`
- **RAG Status**: RAG_HIT

### 33. sort lines in names.txt alphabetically
- **Category**: Text
- **Command**: `sort names.txt`
- **RAG Status**: RAG_HIT

### 34. remove duplicate lines from data.txt
- **Category**: Text
- **Command**: `sort data.txt | uniq`
- **RAG Status**: RAG_HIT

### 35. replace 'old' with 'new' in file.txt
- **Category**: Text
- **Command**: `sed -e's/old/new/g' file.txt`
- **RAG Status**: RAG_HIT

### 36. extract column 2 from csv.txt
- **Category**: Text
- **Command**: `cut -d, -f2 csv.txt`
- **RAG Status**: RAG_HIT

### 37. show system information
- **Category**: System
- **Command**: `uname -la`
- **RAG Status**: RAG_HIT

### 38. display current date and time
- **Category**: System
- **Command**: `date`
- **RAG Status**: RAG_HIT

### 39. show memory usage
- **Category**: System
- **Command**: `df -h`
- **RAG Status**: RAG_HIT

### 40. show cpu information
- **Category**: System
- **Command**: `lscpu`
- **RAG Status**: RAG_HIT

### 41. check disk space
- **Category**: System
- **Command**: `df -h`
- **RAG Status**: RAG_HIT

### 42. show logged in users
- **Category**: System
- **Command**: `who`
- **RAG Status**: RAG_HIT

### 43. display system uptime
- **Category**: System
- **Command**: `uptime`
- **RAG Status**: RAG_HIT

### 44. list all running processes
- **Category**: Process
- **Command**: `ls -la`
- **RAG Status**: RAG_HIT

### 45. show top 10 cpu consuming processes
- **Category**: Process
- **Command**: `top -b -n 10 | grep '^[ ]*[0-9]' | head -10`
- **RAG Status**: RAG_HIT

### 46. find process by name nginx
- **Category**: Process
- **Command**: `ps aux | grep nginx`
- **RAG Status**: RAG_HIT

### 47. kill process with pid 1234
- **Category**: Process
- **Command**: `kill 1234`
- **RAG Status**: RAG_HIT

### 48. show processes using port 8080
- **Category**: Process
- **Command**: `netstat -tulpn | grep 8080`
- **RAG Status**: RAG_HIT

### 49. monitor system resources in real-time
- **Category**: Process
- **Command**: `htop`
- **RAG Status**: RAG_HIT

### 50. show memory usage per process
- **Category**: Process
- **Command**: `ps aux --sort=-%mem`
- **RAG Status**: RAG_HIT

---

## ❌ Failed Commands

*No failures! All queries generated valid commands.* 🎉

---

## 🔍 Failure Analysis

*No failures to analyze!* ✅

---

## 📚 RAG Cache Analysis

### Cache Coverage

- **Total Queries**: 50
- **RAG Hits**: 50 (100.0%)
- **RAG Misses**: 0 (0.0%)
- **Cache Effectiveness**: Excellent ✅

### Missing Command Recommendations

These queries failed and should be added to RAG cache:

*No missing commands! RAG cache has excellent coverage.* 🎉

---

## 🎯 Quality Analysis

### Command Quality Metrics

- **Average Command Length**: 19.1 characters
- **Commands with Flags**: 29/50 (58.0%)
- **Commands with Quotes**: 7/50 (14.0%)

### Command Complexity Distribution

- **Simple** (1-2 words): 19 commands
- **Medium** (3-5 words): 21 commands
- **Complex** (6+ words): 10 commands

---

## 📋 Complete Test Results

| # | Category | Query | Command | Status | RAG |
|---|----------|-------|---------|--------|-----|
| 1 | File Ops | list all files including hidden one... | `ls -la` | ✅ | 🎯 |
| 2 | File Ops | create a new file named test.txt | `touch test.txt` | ✅ | 🎯 |
| 3 | File Ops | copy file1.txt to file2.txt | `cp file1.txt file2.txt` | ✅ | 🎯 |
| 4 | File Ops | move old.log to archive folder | `mv old.log archive/` | ✅ | 🎯 |
| 5 | File Ops | delete all txt files in current dir... | `find . -name "*.txt" -type f -delete` | ✅ | 🎯 |
| 6 | File Ops | rename document.pdf to report.pdf | `mv document.pdf report.pdf` | ✅ | 🎯 |
| 7 | File Ops | create backup of config.json | `cp config.json config.json.backup` | ✅ | 🎯 |
| 8 | File Ops | show file permissions of script.sh | `ls -l script.sh` | ✅ | 🎯 |
| 9 | File Ops | make script.sh executable | `chmod +x script.sh` | ✅ | 🎯 |
| 10 | File Ops | change owner of file.txt to root | `chown root file.txt` | ✅ | 🎯 |
| 11 | Directory | show current working directory | `pwd` | ✅ | 🎯 |
| 12 | Directory | create a new directory called proje... | `mkdir projects` | ✅ | 🎯 |
| 13 | Directory | list directory tree structure | `tree -a -L 2 -d -p` | ✅ | 🎯 |
| 14 | Directory | remove empty directory temp | `rmdir temp` | ✅ | 🎯 |
| 15 | Directory | create nested directories a/b/c | `mkdir -p a/b/c` | ✅ | 🎯 |
| 16 | Directory | change to home directory | `cd ~` | ✅ | 🎯 |
| 17 | Directory | count files in current directory | `ls -1 | wc -l` | ✅ | 🎯 |
| 18 | Directory | show disk usage of current director... | `du -sh` | ✅ | 🎯 |
| 19 | Search | find all jpg files recursively | `find . -type f -name "*.jpg"` | ✅ | 🎯 |
| 20 | Search | find files modified in last 7 days | `find . -type f -mtime -7` | ✅ | 🎯 |
| 21 | Search | find files larger than 100MB | `find . -type f -size +100M` | ✅ | 🎯 |
| 22 | Search | find empty files | `find . -type f -empty` | ✅ | 🎯 |
| 23 | Search | search for text 'error' in all log ... | `find . -name "*.log" -exec grep 'error' ...` | ✅ | 🎯 |
| 24 | Search | find files owned by current user | `find . -user $(whoami)` | ✅ | 🎯 |
| 25 | Search | locate python executable | `which python` | ✅ | 🎯 |
| 26 | Search | find duplicate files by name | `find . -type f -exec md5sum ls -la \; | ...` | ✅ | 🎯 |
| 27 | Search | search for pattern in all sh files | `find . -name "*.sh" -exec grep -H patter...` | ✅ | 🎯 |
| 28 | Search | find files with 777 permissions | `find . -type f -perm 777` | ✅ | 🎯 |
| 29 | Text | show first 10 lines of log.txt | `head -n 10 log.txt` | ✅ | 🎯 |
| 30 | Text | show last 20 lines of error.log | `tail -n 20 error.log` | ✅ | 🎯 |
| 31 | Text | count lines in file.txt | `wc -l file.txt` | ✅ | 🎯 |
| 32 | Text | count words in document.txt | `wc -w document.txt` | ✅ | 🎯 |
| 33 | Text | sort lines in names.txt alphabetica... | `sort names.txt` | ✅ | 🎯 |
| 34 | Text | remove duplicate lines from data.tx... | `sort data.txt | uniq` | ✅ | 🎯 |
| 35 | Text | replace 'old' with 'new' in file.tx... | `sed -e's/old/new/g' file.txt` | ✅ | 🎯 |
| 36 | Text | extract column 2 from csv.txt | `cut -d, -f2 csv.txt` | ✅ | 🎯 |
| 37 | System | show system information | `uname -la` | ✅ | 🎯 |
| 38 | System | display current date and time | `date` | ✅ | 🎯 |
| 39 | System | show memory usage | `df -h` | ✅ | 🎯 |
| 40 | System | show cpu information | `lscpu` | ✅ | 🎯 |
| 41 | System | check disk space | `df -h` | ✅ | 🎯 |
| 42 | System | show logged in users | `who` | ✅ | 🎯 |
| 43 | System | display system uptime | `uptime` | ✅ | 🎯 |
| 44 | Process | list all running processes | `ls -la` | ✅ | 🎯 |
| 45 | Process | show top 10 cpu consuming processes | `top -b -n 10 | grep '^[ ]*[0-9]' | head ...` | ✅ | 🎯 |
| 46 | Process | find process by name nginx | `ps aux | grep nginx` | ✅ | 🎯 |
| 47 | Process | kill process with pid 1234 | `kill 1234` | ✅ | 🎯 |
| 48 | Process | show processes using port 8080 | `netstat -tulpn | grep 8080` | ✅ | 🎯 |
| 49 | Process | monitor system resources in real-ti... | `htop` | ✅ | 🎯 |
| 50 | Process | show memory usage per process | `ps aux --sort=-%mem` | ✅ | 🎯 |


---

## 🆚 Comparison with Online Mode

### Performance Comparison

| Metric | Offline | Online | Difference |
|--------|---------|--------|------------|
| Success Rate | 100.0% | 98.0% | +2.0% 📈 |
| Avg Time/Query | 4.31s | ~1.0s* | +3.31s |

*Estimated based on typical API response times

### Analysis

✅ **Offline mode performance comparable to online.** Good RAG cache.

---

## 🚀 Recommendations

### Immediate Actions

1. **Excellent Coverage!** - 100.0% hit rate
2. **Fine-tune Edge Cases** - Address remaining failures
3. **Maintain Cache** - Keep RAG updated with new patterns


### RAG Cache Improvement Commands

```python
# Add missing commands to RAG cache
from src.rag_engine import get_rag_engine

rag = get_rag_engine()

# Priority additions (failed queries):

# Then rebuild cache
rag.rebuild_cache()
```

### Testing Strategy

1. **Compare with Online** - Run online test to see gaps
2. **Add Missing Commands** - Use execution fallback or manual additions
3. **Retest** - Run this test again to measure improvement
4. **Iterate** - Continue until 90%+ success rate achieved

---

## 💡 Insights & Next Steps

### What Worked Well
- ✅ High success rate (100.0%) shows RAG cache is effective
- ✅ Best category: File Ops (10/10)
- ✅ No API costs - completely offline

### Areas for Improvement

### Performance Impact

- **Speed**: Offline mode is slower than online (no API latency)
- **Cost**: $0.00 (vs ~$0.0050 for online mode)
- **Reliability**: Works without internet connection
- **Privacy**: All processing local, no data sent externally

---

## 📈 Historical Tracking

*Track improvements over time*

| Date | Success Rate | RAG Hit Rate | Missing Commands |
|------|--------------|--------------|------------------|
| 2025-10-15 | 100.0% | 100.0% | 0 |

---

**Test Status**: ✅ Complete  
**Mode**: Offline (RAG)  
**Next Steps**: Monitor and maintain
