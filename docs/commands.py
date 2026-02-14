"""
Command constants for the AI Desktop Assistant.
"""

import re
import threading
import time
from datetime import datetime
import difflib
from typing import Optional, Any

# Command sets
RESET_CHAT = {"/resetchat", "/chatreset", "/clearchat"}
RESET_MEMORY = {"/resetmemory", "/memoryreset", "/clearmemory"}
RESET_REASONING = {"/resetreasoning", "/reasoningreset", "/clearreasoning"}
RESET_META_MEMORY = {"/resetmetamemory", "/metamemoryreset", "/clearmetamemory"}
RESET_ALL = {"/resetall", "/clearall"}

REMOVE_IDENTITY = {"/removeidentity", "/clearidentity", "/deleteidentity"}
REMOVE_FACT = {"/removefact", "/clearfact", "/deletefact", "/removefacts", "/clearfacts"}
REMOVE_PREFERENCE = {"/removepreference", "/clearpreference", "/deletepreference", "/removepreferences", "/clearpreferences"}
REMOVE_GOAL = {"/removegoal", "/cleargoal", "/deletegoal", "/removegoals", "/cleargoals"}
REMOVE_BELIEF = {"/removebelief", "/clearbelief", "/deletebelief", "/removebeliefs", "/clearbeliefs"}
REMOVE_PERMISSION = {"/removepermission", "/clearpermission", "/deletepermission", "/removepermissions", "/clearpermissions"}
REMOVE_RULE = {"/removerule", "/clearrule", "/deleterule", "/removerules", "/clearrules"}
REMOVE_REFUTED = {"/clearrefuted", "/clearrefutedmemories", "/removerefuted", "/deleterefuted", "/clearrefutedbeliefs"}

DOCUMENT_LIST = {"/documents", "/docs", "/listdocs"}
DOCUMENT_REMOVE = {"/removedoc", "/removedocument", "/deletedoc", "/deletedocument"}
DOCUMENT_CONTENT = {"/doccontent", "/docsummarize", "/docpreview"}

NON_LOCKING_COMMANDS = {
    "/status", "/daydreamstatus", "/ddstatus", 
    "/memories", "/chatmemories", "/chatmemory", "/metamemories", "/meta-memories", 
    "/memorystats", "/memorystatistics", 
    "/documents", "/docs", "/listdocs", 
    "/listcommands", "/help", "/commands", "/shadow", "/goals",
    "/specialmemories", "/notes"
}

# Flatten all commands for fuzzy matching
ALL_COMMANDS = set().union(
    RESET_CHAT, RESET_MEMORY, RESET_REASONING, RESET_META_MEMORY, RESET_ALL,
    REMOVE_IDENTITY, REMOVE_FACT, REMOVE_PREFERENCE, REMOVE_GOAL, REMOVE_BELIEF,
    REMOVE_PERMISSION, REMOVE_RULE, REMOVE_REFUTED,
    DOCUMENT_LIST, DOCUMENT_REMOVE, DOCUMENT_CONTENT,
    NON_LOCKING_COMMANDS
)

def handle_command(app: Any, text: str, chat_id: int) -> Optional[str]:
    """Process slash commands and return response if matched"""
    cmd_parts = text.strip().split()
    if not cmd_parts:
        return None
    
    cmd = cmd_parts[0].lower()
    
    # Fuzzy Matching for UX
    if cmd.startswith("/") and cmd not in ALL_COMMANDS and cmd != "/y":
        matches = difflib.get_close_matches(cmd, ALL_COMMANDS, n=1, cutoff=0.8)
        if matches:
            suggestion = matches[0]
            # Auto-correct if very close, or suggest
            # For safety, we'll just suggest for now unless it's a harmless command
            return f"❓ Unknown command '{cmd}'. Did you mean '{suggestion}'?"


    # Confirmation handling
    if cmd == "/y":
        if app.pending_confirmation_command:
            pending_cmd = app.pending_confirmation_command
            app.pending_confirmation_command = None
            
            if pending_cmd in RESET_CHAT:
                if app.current_session_id:
                    app.chat_sessions[app.current_session_id]['history'] = []
                return "♻️ Chat history cleared."

            if pending_cmd in RESET_MEMORY:
                app.memory_store.clear()
                return "🧠 Long-term memory wiped."
                
            if pending_cmd in RESET_REASONING:
                app.reasoning_store.clear()
                return "🧩 Reasoning buffer cleared."

            if pending_cmd in RESET_META_MEMORY:
                app.meta_memory_store.clear()
                return "🧠 Meta-memories cleared."

            if pending_cmd in RESET_ALL:
                if app.current_session_id:
                    app.chat_sessions[app.current_session_id]['history'] = []
                app.reasoning_store.clear()
                app.memory_store.clear()
                app.meta_memory_store.clear()
                return "🔥 Full reset complete (chat + reasoning + memory + meta-memory)."
        else:
            return "ℹ️ No pending command to confirm."

    # Reset Commands (Initiate confirmation)
    if cmd in RESET_CHAT or cmd in RESET_MEMORY or cmd in RESET_REASONING or cmd in RESET_META_MEMORY or cmd in RESET_ALL:
        app.pending_confirmation_command = cmd
        return "⚠️ Are you sure? This action is irreversible. Type `/Y` to confirm."

    # Clear pending confirmation if another command is issued
    app.pending_confirmation_command = None

    # Consolidation
    if cmd in {"/consolidate", "/consolidatenow", "/dream"}:
        def run_consolidation():
            app.log_to_main("🧠 [Binah] Starting manual consolidation...")
            stats = app.binah.consolidate(time_window_hours=None)
            msg = f"🧠 Consolidation complete: Processed {stats['processed']}, Consolidated {stats['consolidated']}, Skipped {stats['skipped']}."
            app.log_to_main(msg)
            app.root.after(0, lambda: app.add_chat_message("System", msg, "incoming"))
            app.root.after(0, app.refresh_database_view)
        
        threading.Thread(target=run_consolidation, daemon=True).start()
        return "⏳ Consolidation started in background..."

    # Memory Removal
    if cmd in REMOVE_IDENTITY:
        active_count = len(app.memory_store.get_active_by_type("IDENTITY"))
        count = app.memory_store.clear_by_type("IDENTITY")
        msg = f"🗑️ Removed {count} IDENTITY memories."
        if count > active_count: msg += f" ({active_count} active, {count - active_count} hidden)"
        return msg
    
    if cmd in REMOVE_FACT:
        active_count = len(app.memory_store.get_active_by_type("FACT"))
        count = app.memory_store.clear_by_type("FACT")
        msg = f"🗑️ Removed {count} FACT memories."
        if count > active_count: msg += f" ({active_count} active, {count - active_count} hidden)"
        return msg
        
    if cmd in REMOVE_PREFERENCE:
        active_count = len(app.memory_store.get_active_by_type("PREFERENCE"))
        count = app.memory_store.clear_by_type("PREFERENCE")
        msg = f"🗑️ Removed {count} PREFERENCE memories."
        if count > active_count: msg += f" ({active_count} active, {count - active_count} hidden)"
        return msg
        
    if cmd in REMOVE_GOAL:
        active_count = len(app.memory_store.get_active_by_type("GOAL"))
        count = app.memory_store.clear_by_type("GOAL")
        msg = f"🗑️ Removed {count} GOAL memories."
        if count > active_count: msg += f" ({active_count} active, {count - active_count} hidden)"
        return msg
        
    if cmd in REMOVE_BELIEF:
        active_count = len(app.memory_store.get_active_by_type("BELIEF"))
        count = app.memory_store.clear_by_type("BELIEF")
        msg = f"🗑️ Removed {count} BELIEF memories."
        if count > active_count: msg += f" ({active_count} active, {count - active_count} hidden)"
        return msg
        
    if cmd in REMOVE_PERMISSION:
        active_count = len(app.memory_store.get_active_by_type("PERMISSION"))
        count = app.memory_store.clear_by_type("PERMISSION")
        msg = f"🗑️ Removed {count} PERMISSION memories."
        if count > active_count: msg += f" ({active_count} active, {count - active_count} hidden)"
        return msg
        
    if cmd in REMOVE_RULE:
        active_count = len(app.memory_store.get_active_by_type("RULE"))
        count = app.memory_store.clear_by_type("RULE")
        msg = f"🗑️ Removed {count} RULE memories."
        if count > active_count: msg += f" ({active_count} active, {count - active_count} hidden)"
        return msg
        
    if cmd in REMOVE_REFUTED:
        active_count = len(app.memory_store.get_active_by_type("REFUTED_BELIEF"))
        count = app.memory_store.clear_by_type("REFUTED_BELIEF")
        msg = f"🗑️ Removed {count} REFUTED_BELIEF memories."
        if count > active_count: msg += f" ({active_count} active, {count - active_count} hidden)"
        return msg

    # Document Management
    if cmd in DOCUMENT_LIST:
        docs = app.document_store.list_documents(limit=20)
        if not docs:
            return "📚 No documents in the database."
        
        lines = []
        for doc_id, filename, file_type, page_count, chunk_count, created_at in docs:
            date_str = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M")
            page_info = f", {page_count} pages" if page_count else ""
            lines.append(f"📄 {filename} ({file_type}{page_info}, {chunk_count} chunks) - {date_str}")
        
        return "📚 Document Database:\n" + "\n".join(lines)

    if cmd in DOCUMENT_REMOVE:
        # Extract filename
        match = re.search(r'"([^"]*)"', text)
        if match:
            doc_filename = match.group(1)
            # Find doc ID
            docs = app.document_store.list_documents(limit=1000)
            doc_id = next((d[0] for d in docs if d[1] == doc_filename), None)
            
            if doc_id:
                if app.document_store.delete_document(doc_id):
                    # Refresh GUI if open
                    app.root.after(0, app.refresh_documents)
                    return f"🗑️ Successfully removed document: {doc_filename}"
                else:
                    return f"❌ Could not remove document: {doc_filename}"
            else:
                return f"❌ Document not found: {doc_filename}"
        else:
            return "🗑️ To remove a document, use: /RemoveDoc \"filename.pdf\"\nUse /Documents to see available documents."

    if cmd in DOCUMENT_CONTENT or any(text.lower().startswith(x) for x in DOCUMENT_CONTENT):
            # Extract filename
        match = re.search(r'"([^"]*)"', text)
        if match:
            doc_filename = match.group(1)
            # Find doc ID
            docs = app.document_store.list_documents(limit=1000)
            doc_id = next((d[0] for d in docs if d[1] == doc_filename), None)
            
            if doc_id:
                chunks = app.document_store.get_document_chunks(doc_id)
                if chunks:
                    preview = "\n\n".join([f"Chunk {c['chunk_index']+1}: {c['text'][:200]}..." for c in chunks[:3]])
                    return f"📖 Content preview for '{doc_filename}':\n\n{preview}"
                return f"❌ No content found for: {doc_filename}"
            return f"❌ Document not found: {doc_filename}"
        else:
            return "📖 To view document content, use: /DocContent \"filename.pdf\"\nUse /Documents to see available documents."

    # Memories View
    if cmd == "/memories":
        items = app.memory_store.list_recent(limit=None)
        if not items:
            return "🧠 No saved memories."
        
        type_emoji = {
            "IDENTITY": "👤", "FACT": "📌", "PREFERENCE": "❤️", 
            "GOAL": "🎯", "RULE": "⚖️", "PERMISSION": "✅", "BELIEF": "💭"
        }
        
        grouped = {}
        for item in items:
            _id, mem_type, subject, text = item[:4]
            grouped.setdefault(mem_type, []).append((subject, text))
        
        lines = []
        hierarchy = ["PERMISSION", "RULE", "IDENTITY", "PREFERENCE", "GOAL", "FACT", "BELIEF"]
        
        for mem_type in hierarchy:
            if mem_type in grouped:
                emoji = type_emoji.get(mem_type, "💡")
                lines.append(f"\n{emoji} {mem_type}:")
                for subject, text in grouped[mem_type]:
                    lines.append(f"  - [{subject}] {text}")
                del grouped[mem_type]
        
        for mem_type, remaining in grouped.items():
            emoji = type_emoji.get(mem_type, "💡")
            lines.append(f"\n{emoji} {mem_type}:")
            for subject, text in remaining:
                lines.append(f"  - [{subject}] {text}")
        
        return "🧠 Saved Memories :\n" + "\n".join(lines)

    # Meta Memories View
    if cmd in {"/metamemories", "/meta-memories"}:
        items = app.meta_memory_store.list_recent(limit=30)
        if not items:
            return "🧠 No meta-memories."
        
        lines = []
        for item in items:
            _id = item[0]
            event_type = item[1]
            subject = item[2]
            text = item[3]
            created_at = item[4]
            
            event_emoji = {
                "MEMORY_CREATED": "✨", "VERSION_UPDATE": "🔄",
                "CONFLICT_DETECTED": "⚠️", "CONSOLIDATION": "🔗"
            }.get(event_type, "🧠")
            lines.append(f"{event_emoji} [{subject}] {text}")
        
        return "🧠 Meta-Memories (Reflections):\n" + "\n".join(lines)

    # Shadow Memory View (Adversarial Error Log)
    if cmd in {"/shadow", "/shadowmemory", "/errors"}:
        if hasattr(app.memory_store, 'get_shadow_memories'):
            items = app.memory_store.get_shadow_memories(limit=10)
            if not items:
                return "🌑 No shadow memories (mistakes) found."
            
            lines = []
            for item in items:
                date_str = datetime.fromtimestamp(item['created_at']).strftime("%Y-%m-%d %H:%M")
                lines.append(f"[{date_str}] ⚠️ {item['text'][:50]}... -> {item['reason']}")
            return "🌑 Shadow Memory (Recent Failures):\n" + "\n".join(lines)
        return "❌ Memory store does not support shadow memory."

    # Goals View
    if cmd in {"/goals", "/activegoals"}:
        goals = app.memory_store.get_active_by_type("GOAL")
        if not goals:
            return "🎯 No active goals."
        
        lines = [f"🎯 Active Goals ({len(goals)}):"]
        for g in goals:
            # g: (id, subject, text, source, confidence, progress)
            progress = g[5] if len(g) > 5 else 0.0
            lines.append(f"- [ID:{g[0]}] {g[2]} ({int(progress*100)}%)")
        return "\n".join(lines)

    # Chat Memories View
    if cmd in {"/chatmemories", "/chatmemory"}:
        items = app.memory_store.list_recent(limit=None)
        if not items:
            return "🧠 No saved memories."
        
        # Filter out daydream memories
        chat_items = [item for item in items if len(item) >= 5 and item[4] != 'daydream']
        
        if not chat_items:
            return "🧠 No chat memories found."

        type_emoji = {
            "IDENTITY": "👤", "FACT": "📌", "PREFERENCE": "❤️", 
            "GOAL": "🎯", "RULE": "⚖️", "PERMISSION": "✅", "BELIEF": "💭"
        }
        
        grouped = {}
        for item in chat_items:
            _id, mem_type, subject, text = item[:4]
            grouped.setdefault(mem_type, []).append((subject, text))
        
        lines = []
        hierarchy = ["PERMISSION", "RULE", "IDENTITY", "PREFERENCE", "GOAL", "FACT", "BELIEF"]
        
        for mem_type in hierarchy:
            if mem_type in grouped:
                emoji = type_emoji.get(mem_type, "💡")
                lines.append(f"\n{emoji} {mem_type}:")
                for subject, text in grouped[mem_type]:
                    lines.append(f"  - [{subject}] {text}")
                del grouped[mem_type]
        
        for mem_type, remaining in grouped.items():
            emoji = type_emoji.get(mem_type, "💡")
            lines.append(f"\n{emoji} {mem_type}:")
            for subject, text in remaining:
                lines.append(f"  - [{subject}] {text}")
        
        return "🧠 Chat Memories (No Daydreams):\n" + "\n".join(lines)

    # Assistant Notes (formerly Special Memories)
    if cmd in {"/note", "/notes", "/specialmemory", "/specialmemories", "/chronicle", "/chronicles"}:
        # If arguments provided, create note
        if len(cmd_parts) > 1:
            content = text[len(cmd_parts[0]):].strip()
            if app.decider:
                app.decider.create_note(content)
                return f"📝 Note created: {content}"
            else:
                return "❌ Decider not initialized."
        
        # List notes
        items = app.memory_store.list_recent(limit=None)
        if not items:
            return "🧠 No saved memories."
        
        notes = [item for item in items if item[1] == "NOTE"]
        
        if not notes:
            return "📝 No assistant notes found."
        
        lines = []
        for item in notes:
            # item: (id, type, subject, text, source, verified)
            _id, mem_type, subject, text = item[:4]
            lines.append(f"📝 [ID:{_id}] {text}")
        
        return "📝 Assistant Notes:\n" + "\n".join(lines)

    if cmd in {"/clearnotes", "/clearspecialmemory"}:
        items = app.memory_store.list_recent(limit=None)
        count = 0
        for item in items:
            if item[1] == "NOTE":
                if app.memory_store.delete_entry(item[0]):
                    count += 1
        return f"📝 Cleared {count} notes."

    # Remove Summaries
    if cmd in {"/removesummaries", "/clearsummaries", "/deletesummaries"}:
        if not app.meta_memory_store:
            return "❌ Meta-memory store not initialized."
        
        count_summary = app.meta_memory_store.delete_by_event_type("SESSION_SUMMARY")
        count_analysis = app.meta_memory_store.delete_by_event_type("HOD_ANALYSIS")
        total = count_summary + count_analysis
        
        # Refresh UI if needed
        app.root.after(0, app.refresh_database_view)
        
        return f"🗑️ Removed {total} summaries ({count_summary} session summaries, {count_analysis} Hod analyses)."

    # Consolidate Summaries
    if cmd in {"/consolidatesummaries", "/compresssummaries"}:
        if not app.daat:
            return "❌ Da'at not initialized."
        
        result = app.daat.consolidate_summaries()
        app.root.after(0, app.refresh_database_view)
        return result

    # Status
    if cmd == "/status":
        status_msg = "📊 **System Status**\n\n"
        status_msg += f"🔌 Telegram Bridge: {'Connected' if app.is_connected() else 'Disconnected'}\n"
        
        cycle_limit = int(app.settings.get("daydream_cycle_limit", 15))
        cycle_info = f"(Cycle {app.daydream_cycle_count}/{cycle_limit})"
        
        status_msg += f"🤖 AI Mode: ☁️ Daydream Mode (Active) {cycle_info}\n"
        status_msg += f"⚙️ Processing: {'⏳ Busy' if app.is_processing else '✅ Idle'}\n"
        status_msg += f"📚 Knowledge Base: {app.document_store.get_total_documents()} files ({app.document_store.get_total_chunks()} chunks)\n"
        
        mem_items = app.memory_store.list_recent(limit=None)
        verified_count = sum(1 for item in mem_items if len(item) > 5 and item[5] == 1)
        status_msg += f"🧠 Memory: {len(mem_items)} active nodes ({verified_count} verified)\n"
        
        if app.keter:
            keter_stats = app.keter.evaluate()
            status_msg += f"👑 Keter Coherence: {keter_stats.get('keter', 0.0):.4f}\n"
            
        return status_msg

    # Memory Statistics
    if cmd in {"/memorystatistics", "/memorystats"}:
        items = app.memory_store.list_recent(limit=None)
        if not items: return "📊 Memory is empty."
        
        by_type = {}
        by_source = {}
        verified_count = 0
        for item in items:
            mtype, source, is_verified = item[1], item[4], (item[5] if len(item) > 5 else 0)
            by_type[mtype] = by_type.get(mtype, 0) + 1
            by_source[source] = by_source.get(source, 0) + 1
            if is_verified: verified_count += 1
        
        stats = f"📊 **Memory Statistics**\n\n**Total:** {len(items)}\n**Verified:** {verified_count} ({verified_count/len(items)*100:.1f}%)\n\n**By Type:**\n" + "\n".join([f"- {t}: {c}" for t, c in sorted(by_type.items(), key=lambda x: x[1], reverse=True)]) + "\n\n**By Source:**\n" + "\n".join([f"- {s}: {c}" for s, c in sorted(by_source.items(), key=lambda x: x[1], reverse=True)])
        return stats

    # Daydream Status
    if cmd in {"/daydreamstatus", "/ddstatus"}:
        cycle_limit = int(app.settings.get("daydream_cycle_limit", 15))
        status_msg = "☁️ **Daydream Status**\n\n"
        
        if not app.decider:
            status_msg += "❌ State: Not Initialized\n"
        else:
            status_msg += f"✅ State: {'Processing' if app.is_processing else 'Active (Idle loop)'}\n"
            
        status_msg += f"🔄 Cycle Progress: {app.daydream_cycle_count} / {cycle_limit}\n"
        return status_msg

    # Verification
    if cmd in {"/verifysources", "/verify"}:
        app.root.after(1000, app.verify_memory_sources)
        return "🕵️ Source verification scheduled."

    if cmd in {"/verifyall", "/verifyallsources"}:
        app.root.after(1000, app.verify_all_memory_sources)
        return "🕵️ Full verification loop scheduled."

    if cmd == "/stop":
        app.stop_processing()
        return "🛑 All processing stopped."

    if cmd in {"/stopverifying", "/stopverify"}:
        app.stop_processing()
        return "🛑 Verification stopped."
        
    if cmd == "/terminate_desktop":
        app.root.after(1000, app.root.destroy)
        return "👋 Shutting down desktop assistant..."

    # Decider Commands
    if cmd == "/decider":
        if len(cmd_parts) < 2:
            return "🤖 Decider Usage: /decider [up|down|daydream|verify|verifyall|loop|stopdaydream]"
        
        action = cmd_parts[1].lower()
        if not app.decider:
            return "❌ Decider not initialized."

        if action == "up":
            app.decider.increase_temperature()
            return "🌡️ Temperature increased."
        elif action == "down":
            app.decider.decrease_temperature()
            return "🌡️ Temperature decreased."
        elif action == "daydream":
            app.decider.start_daydream()
            return "☁️ Daydream triggered."
        elif action == "verify":
            app.decider.start_verification_batch()
            return "🕵️ Verification triggered."
        elif action == "verifyall":
            app.decider.verify_all()
            return "🕵️ Full verification triggered."
        elif action == "loop":
            app.decider.start_daydream_loop()
            return "🔄 Daydream loop enabled."
        elif action == "stopdaydream":
            app.decider.stop_daydream()
            return "🛑 Daydream stopped."
        else:
            return f"❌ Unknown decider action: {action}"

    # List Commands
    if cmd in {"/listcommands", "/help", "/commands"}:
        return (
            "🛠️ **Command List**\n\n"
            "**System:**\n"
            "• `/Status` - Show system state\n"
            "• `/DaydreamStatus` - Show daydream cycle info\n"
            "• `/Disrupt` - Interrupt current loop (Telegram only)\n"
            "• `/Stop` - Stop ALL processing (Chat, Docs, Verify)\n"
            "• `/StopVerifying` - Stop verification loop\n"
            "• `/Terminate_Desktop` - Close application\n\n"
            
            "**Memory:**\n"
            "• `/Memories` - Show all memories\n"
            "• `/ChatMemories` - Show chat memories\n"
            "• `/MetaMemories` - Show memory logs\n"
            "• `/MemoryStats` - Show memory counts\n"
            "• `/Shadow` - Show recent errors\n"
            "• `/Goals` - Show active goals\n"
            "• `/Consolidate` - Merge duplicates\n"
            "• `/SpecialMemories` - Show special memories\n"
            "• `/SpecialMemory [text]` - Add special memory\n"
            "• `/ClearSpecialMemory` - Clear all special memories\n"
            "• `/Verify` - Verify sources (batch)\n"
            "• `/VerifyAll` - Verify all sources\n\n"
            
            "**Docs:**\n"
            "• `/Documents` - List files\n"
            "• `/DocContent \"file\"` - Read file\n"
            "• `/RemoveDoc \"file\"` - Delete file\n\n"
            
            "**Cleanup:**\n"
            "• `/ResetChat` - Clear chat\n"
            "• `/ResetMemory` - Wipe DB\n"
            "• `/Remove[Type]` - Delete type (e.g. /RemoveGoal)\n"
            "• `/ClearRefutedMemories` - Delete all refuted beliefs"
        )

    return None