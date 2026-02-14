# E:/CodingWorkspace/Projects/AITelegramIntegration/event_bus.py
from typing import Callable, Dict, List, Any, Tuple
from dataclasses import dataclass, field
import queue
import threading
import itertools
import time
import concurrent.futures
import logging


@dataclass
class Event:
    """Standard event packet."""
    type: str
    data: Any = None
    source: str = "System"
    priority: int = 0
    timestamp: float = field(default_factory=time.time)
    ttl: float = 5.0  # Seconds before event is considered stale

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class EventBus:
    """
    Central nervous system for the architecture.
    Decouples components by allowing them to publish/subscribe to events.
    Supports prioritization and activity logging.
    
    Refactored to use an asynchronous priority queue to prevent recursive event cascades.
    """

    def __init__(self, log_fn: Callable[[str], None] = logging.info, executor: concurrent.futures.Executor = None):
        self.log = log_fn
        # Subscribers: Dict[event_type, List[(priority, callback)]]
        self._subscribers: Dict[str, List[Tuple[int, Callable[[Event], None]]]] = {}
        self._history: List[Event] = []
        self._dead_letter_queue: List[Tuple[Event, str]] = [] # (Event, Error)
        self._history_limit = 1000
        
        # Load Management
        self._event_budget = 100  # Max events per second
        self._event_counts = []   # Sliding window of timestamps
        self._storm_threshold = 50 # Events per second to trigger storm warning
        self._load_shedding_threshold = 200 # Queue size to start dropping low priority
        self._lock = threading.Lock() # Thread safety for rate limiting
        self._last_storm_warning = 0.0 # Cooldown for storm warnings
        
        # Metrics
        self._metrics = {
            "dropped_events": 0,
            "stale_events": 0,
            "processed_events": 0,
            "storm_events": 0,
            "total_latency": 0.0,
        }
        
        # Priority Queue: Stores (-priority, counter, event)
        self._queue = queue.PriorityQueue()
        self._counter = itertools.count()
        self._running = True
        self._executor = executor if executor else concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="EventBusSub")
        self._owns_executor = executor is None
        self._thread = threading.Thread(target=self._process_loop, daemon=True, name="EventBusWorker")
        self._thread.start()

    def subscribe(self, event_type: str, callback: Callable[[Event], None], priority: int = 0):
        """
        Register a callback for a specific event type.
        Priority: Higher values run first. Default 0.
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append((priority, callback))
        # Sort by priority descending (High priority first)
        self._subscribers[event_type].sort(key=lambda x: x[0], reverse=True)

    def publish(self, event_type: str, data: Any = None, source: str = "System", priority: int = 0, ttl: float = 5.0):
        """
        Schedule an event for processing.
        Puts the event into the priority queue.
        """
        now = time.time()
        
        with self._lock:
            # 1. Rate Limiting / Budget Check
            # Remove timestamps older than 1 second
            self._event_counts = [t for t in self._event_counts if now - t < 1.0]
            
            if len(self._event_counts) >= self._event_budget:
                # Budget exceeded: Drop low priority events
                if priority < 10: # Critical threshold
                    # logging.warning(f"⚠️ EventBus: Rate limit exceeded. Dropping '{event_type}' (Priority {priority})")
                    
                    self._metrics["dropped_events"] += 1
                    return
            
            self._event_counts.append(now)
            current_rate = len(self._event_counts)
        
        # 2. Load Shedding (Priority Suppression)
        q_size = self._queue.qsize()
        if q_size > self._load_shedding_threshold:
            # If queue is backed up, only accept high priority
            if priority < 5:
                # logging.warning(f"⚠️ EventBus: High load ({q_size}). Dropping '{event_type}'")
                
                self._metrics["dropped_events"] += 1
                return

        # 3. Storm Detection (Simple heuristic)
        if current_rate > self._storm_threshold and (now - self._last_storm_warning > 1.0):
             # Active Storm Detection: Signal the system
             # Use direct queue put to avoid recursion/rate limits on the warning itself
             storm_event = Event("EVENT_STORM_DETECTED", data={"rate": current_rate}, source="EventBus", priority=100, ttl=1.0)
             self._queue.put((-100, next(self._counter), storm_event))
             
             self._metrics["storm_events"] += 1
             self._last_storm_warning = now

        event = Event(type=event_type, data=data, source=source, priority=priority, ttl=ttl)

        # Enqueue with priority (negated for Min-Heap) and counter for stability
        count = next(self._counter)
        self._queue.put((-priority, count, event))

    def _process_loop(self):
        """Background worker to process events sequentially."""
        while self._running:
            try:
                # Get next event (blocking with timeout)
                _, _, event = self._queue.get(timeout=1.0)
                
                # 4. TTL Check (Stale Events)
                if time.time() - event.timestamp > event.ttl:
                    # logging.warning(f"🗑️ EventBus: Dropping stale event '{event.type}' (Age: {time.time() - event.timestamp:.2f}s)")
                    
                    self._metrics["stale_events"] += 1
                    self._queue.task_done()
                    continue

                self._dispatch(event)
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.log(f"❌ EventBus Loop Error: {e}")

    def _dispatch(self, event: Event):
        """Execute callbacks for a single event."""
        with self._lock:
            # Log to history (Option B: Actual Dispatch)
            self._history.append(event)
            if len(self._history) > self._history_limit:
                self._history.pop(0)

            # Calculate latency
            dispatch_delay = time.time() - event.timestamp
            self._metrics["processed_events"] += 1
            self._metrics["total_latency"] += dispatch_delay

        # Notify specific listeners
        if event.type in self._subscribers:
            for _, callback in self._subscribers[event.type]:
                self._executor.submit(self._safe_callback_run, callback, event)

        # Notify wildcard listeners
        if "*" in self._subscribers:
            for _, callback in self._subscribers["*"]:
                self._executor.submit(self._safe_callback_run, callback, event)

    def _safe_callback_run(self, callback, event):
        """Execute callback safely within thread pool."""
        try:
            callback(event)
        except Exception as e:
            self.log(f"❌ Event Bus Error processing '{event.type}': {e}")
            self._dead_letter_queue.append((event, str(e)))

    def get_history(self, limit: int = 100) -> List[Event]:
        """Get recent event history for visualization."""
        return self._history[-limit:]

    def get_dead_letters(self) -> List[Tuple[Event, str]]:
        """Get failed events."""
        return list(self._dead_letter_queue)

    def stop(self):
        """Stop the processing thread."""
        self._running = False
        if self._owns_executor:
            self._executor.shutdown(wait=False)
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
            
    def get_load(self) -> int:
        """Return current queue size."""
        return self._queue.qsize()

    def get_metrics(self) -> Dict[str, Any]:
        """Return current operational metrics."""
        with self._lock:
            metrics = self._metrics.copy()
            processed = metrics.get("processed_events", 0)
            total_latency = metrics.get("total_latency", 0.0)
            
            metrics["avg_latency"] = total_latency / processed if processed > 0 else 0.0
            return metrics