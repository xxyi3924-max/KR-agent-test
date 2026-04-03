"""
Scheduler — Runs agent multiple times per day
"""

import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Callable, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ScheduleConfig:
    """Agent schedule configuration"""
    run_interval_hours: float = 6      # Run every 6 hours
    check_interval_minutes: int = 5    # Check for fills between runs
    signal_refresh_days: int = 3       # Refresh signals every N days
    max_consecutive_errors: int = 3    # Stop after N errors


class AgentScheduler:
    """Schedules and runs the agent at configured intervals"""
    
    def __init__(self, agent_func: Callable, config: ScheduleConfig):
        self.agent_func = agent_func
        self.config = config
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.last_run: Optional[datetime] = None
        self.last_signal_refresh: Optional[datetime] = None
        self.consecutive_errors = 0
        
    def start(self):
        """Start the scheduler"""
        if self.running:
            logger.warning("Scheduler already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logger.info("Scheduler started")
    
    def stop(self):
        """Stop the scheduler"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=10)
        logger.info("Scheduler stopped")
    
    def _run_loop(self):
        """Main scheduler loop"""
        self.last_run = datetime.now()
        self.last_signal_refresh = datetime.now()
        
        while self.running:
            try:
                # Calculate next run time
                next_run = self.last_run + timedelta(hours=self.config.run_interval_hours)
                now = datetime.now()
                
                # Check if time for main run
                if now >= next_run:
                    self._execute_run()
                    self.last_run = now
                    
                    # Check if need to refresh signals
                    signal_age = (now - self.last_signal_refresh).days
                    if signal_age >= self.config.signal_refresh_days:
                        logger.info("Refreshing TDash signals...")
                        self.last_signal_refresh = now
                
                # Background checks between runs
                self._background_check()
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                self.consecutive_errors += 1
                
                if self.consecutive_errors >= self.config.max_consecutive_errors:
                    logger.critical(f"Max errors reached ({self.config.max_consecutive_errors}), stopping")
                    self.running = False
                    break
            
            # Sleep before next check
            time.sleep(self.config.check_interval_minutes * 60)
    
    def _execute_run(self):
        """Execute a full agent run"""
        logger.info(f"=== Agent Run Started: {datetime.now().isoformat()} ===")
        
        try:
            summary = self.agent_func()
            logger.info(f"Run complete: {summary}")
            self.consecutive_errors = 0
        except Exception as e:
            logger.error(f"Agent run failed: {e}")
            raise
    
    def _background_check(self):
        """Lightweight checks between main runs"""
        # Could check order fills, update prices, etc.
        # For now, just log
        logger.debug("Background check: all systems nominal")
    
    def force_run(self):
        """Force an immediate run"""
        logger.info("Force run triggered")
        self._execute_run()
        self.last_run = datetime.now()
    
    def get_status(self) -> dict:
        """Get scheduler status"""
        return {
            "running": self.running,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": (
                self.last_run + timedelta(hours=self.config.run_interval_hours)
            ).isoformat() if self.last_run else None,
            "consecutive_errors": self.consecutive_errors
        }


class SimpleTimer:
    """Simple timer for one-off delayed execution"""
    
    def __init__(self, delay_seconds: float, callback: Callable):
        self.delay = delay_seconds
        self.callback = callback
        self.thread: Optional[threading.Thread] = None
        
    def start(self):
        """Start the timer"""
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
    
    def _run(self):
        """Wait and execute"""
        time.sleep(self.delay)
        try:
            self.callback()
        except Exception as e:
            logger.error(f"Timer callback failed: {e}")
