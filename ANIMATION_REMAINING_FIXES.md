# 2D Animation System - Remaining Fixes

## ✅ What Works
- Backend creates jobs instantly (0.002s response)
- Animation server generates sprites successfully
- Uploads to Supabase completed
- Beautiful UI components created

## ❌ Issues Found & Fixes Needed

### 1. Frontend API URL Wrong (CRITICAL)

**Problem:**
```
API Base: http://127.0.0.1:8000  ❌
Should be: http://127.0.0.1:5050 ✅
```

**Location:** `editor/docks/ai_chat_dock.cpp:14509`

**Current Code:**
```cpp
animation_tracker->initialize(api_endpoint.replace("/chat", ""), this);
```

**Fix Needed:**
```cpp
// Use _get_api_base_url() instead to get correct port
String anim_api_base = _get_api_base_url();
animation_tracker->initialize(anim_api_base, this);
```

**Why:** `api_endpoint` is set during ENTER_TREE which might use old/wrong port. The `_get_api_base_url()` function correctly determines dev vs prod.

---

### 2. Animations Not Assigned to Characters

**Problem:**
```
📊 Loaded project: 16bit_viking_axe_swing
   Characters: 0  ❌
   Animations: 2
   ⚠️  2 animations not assigned to any character
```

**Root Cause:** Backend doesn't create character entries when uploading animations.

**Location:** `backend/sprite_sheet_gen/supabase_storage.py` (probably)

**Fix Needed:**
When uploading animation project:
1. Create character entry in `characters` table
2. Link animations to character via `character_id`
3. Set `visual_traits` and `character_type`

---

### 3. add_branch Endpoint 404

**Problem:**
```
404 Client Error: Not Found for url: http://127.0.0.1:8001/add_branch/{project_id}
```

**Check:** `backend/sprite_sheet_gen/animation_server.py`

**Current Endpoint:**
```python
@app.post("/add_branch/{job_id}")  # Uses job_id, not project_id!
```

**Fix Needed:**
Either:
- Change endpoint to accept project_id: `@app.post("/add_branch/{project_id}")`
- Or change backend call to use job_id instead

---

### 4. Job Completion Not Logged

**Problem:** Animation server stops after Supabase upload without final message

**Location:** `backend/sprite_sheet_gen/animation_server.py` - `create_and_execute_animation_graph_job()`

**Fix Needed:**
Add at end of function:
```python
except Exception as e:
    JOBS[job_id]["status"] = "failed"
    JOBS[job_id]["error"] = str(e)
    JOBS[job_id]["updated_at"] = datetime.now().isoformat()
    print(f"Job {job_id}: Failed - {e}")
finally:
    print(f"✅ Job {job_id}: Background task completed (check status above)")
```

---

### 5. Edit Operation Can't Find Animations

**Problem:**
```
"vikingwalk": {"error": "Animation 'viking_walk' not found", "success": false}
```

**Root Cause:** Likely character assignment issue - animations exist in DB but not linked properly

**Fix:** Same as Issue #2 - need proper character creation

---

## 🚀 Quick Fixes (Priority Order)

### **Fix 1: Frontend API URL (5 minutes)**
```cpp
// In ai_chat_dock.cpp, line ~14509
String anim_api_base = _get_api_base_url();
animation_tracker->initialize(anim_api_base, this);
```

### **Fix 2: add_branch Endpoint (2 minutes)**
```python
# In animation_server.py, change line ~649
@app.post("/add_branch/{project_id}")  # Changed from job_id
async def add_animation_branch(
    project_id: str,  # Changed from job_id
    request: AddBranchRequest,
    ...
```

### **Fix 3: Character Creation (15 minutes)**
Need to modify Supabase upload logic to:
1. Create character from project_name
2. Link all animations to that character
3. Set character_type = "character"

---

## ⚡ Immediate Workaround

**For Testing NOW** (without rebuilding):

1. **Test with list_my_animations:**
   ```
   "Show all my sprite animations"
   ```
   This should work since it queries Supabase directly

2. **Manual Supabase Fix:**
   - Open Supabase dashboard
   - Create character entry manually
   - Link existing animations to character_id

---

## 📝 Files That Need Changes

1. **editor/docks/ai_chat_dock.cpp** - Fix API URL (line 14509)
2. **backend/sprite_sheet_gen/animation_server.py** - Fix add_branch endpoint
3. **backend/sprite_sheet_gen/supabase_storage.py** - Add character creation logic

---

## 🎯 After Fixes

The system will be **fully functional** with:
- ✅ Correct polling (every 15s)
- ✅ Beautiful auto-updating UI
- ✅ Character-based organization
- ✅ Working edit operations
- ✅ add_branch functionality

**Total fix time: ~25 minutes of coding**

---

## Current Status

**Working:**
- Job creation (instant!)
- Video generation
- Supabase upload
- Backend caching
- Numbered references

**Broken:**
- Frontend polling (wrong URL)
- Character assignment
- Edit operations (can't find anims)
- add_branch endpoint

**Next Step:** Fix the API URL first, then test polling!

