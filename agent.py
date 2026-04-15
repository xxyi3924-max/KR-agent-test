"""
Agent Trader - Main Entry Point
Runs scheduled decision cycles using LLM for signal decisions
"""

import os
import logging
import argparse
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler

from decision_engine import DecisionEngine
from etf_monitor import ETFMonitor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('agent_trader.log')
    ]
)
logger = logging.getLogger(__name__)


class Agent:
    """Main agent class that orchestrates trading decisions"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.engine = DecisionEngine(config_path)
        # Wire the real-time quant monitor into the engine so that
        # _manage_free_cash() defers to it when the monitor is running.
        self.etf_monitor = ETFMonitor(self.engine.broker, self.engine.config)
        self.engine.etf_monitor = self.etf_monitor
        logger.info("Agent initialized")
    
    def run_once(self) -> dict:
        """Run a single decision cycle"""
        logger.info("=" * 60)
        logger.info(f"Starting decision cycle at {datetime.now().isoformat()}")
        logger.info("=" * 60)
        
        try:
            summary = self.engine.run_cycle()
            self._log_summary(summary)
            return summary
        except Exception as e:
            logger.error(f"Agent cycle failed: {e}")
            return {"error": str(e)}
    
    def _log_summary(self, summary: dict):
        """Log execution summary"""
        logger.info("-" * 40)
        logger.info("CYCLE SUMMARY")
        logger.info("-" * 40)

        if summary.get('skipped'):
            logger.info(f"Skipped: {summary['skipped']}")

        # Signals
        signals = summary.get('signals', {})
        logger.info(
            f"Signals: {signals.get('total', 0)} total, "
            f"{signals.get('actionable', 0)} actionable "
            f"({signals.get('actions', [])})"
        )
        
        # Plans
        logger.info(f"Plans Created: {summary.get('plans_created', 0)}")
        
        # Executions
        for ex in summary.get('executions', []):
            status = "✅" if ex.get('success') else "❌"
            logger.info(f"{status} {ex.get('action')} {ex.get('ticker')}: {ex.get('message')}")
        
        # Auto-invest
        for inv in summary.get('auto_invest', []):
            logger.info(f"💰 Auto-invest: {inv.get('qty'):.2f} {inv.get('ticker')} @ ${inv.get('price'):.2f}")
        
        # Liquidations
        for liq in summary.get('cash_liquidations', []):
            logger.info(f"📤 Liquidated: {liq.get('qty')} {liq.get('ticker')} → raised ${liq.get('raised'):.2f}")
        
        # Errors
        if summary.get('errors'):
            logger.error(f"Errors: {summary['errors']}")
        
        logger.info("-" * 40)
    
    def start_scheduled(self):
        """Start agent with scheduled runs (4x/day) + 5-min background checks."""
        logger.info("Starting scheduled agent (4x/day + 5-min background checks)")

        # Start real-time ETF quant monitor (daemon thread, polls every 10 s)
        self.etf_monitor.start()

        # Run immediately
        self.run_once()

        scheduler = BlockingScheduler()

        # Main decision cycle — 4x per day
        scheduler.add_job(
            self.run_once,
            'cron',
            hour='0,6,12,18',
            minute=0,
        )

        # Background check every 5 minutes:
        #   - detect limit order fills → place TP/SL
        #   - escalate unfilled limits to market after 1 hour
        #   - cancel orphaned TP/SL legs (OCO simulation)
        scheduler.add_job(
            self.engine.background_check,
            'interval',
            minutes=5,
        )

        try:
            scheduler.start()
        except KeyboardInterrupt:
            logger.info("Agent stopped by user")
            self.etf_monitor.stop()
            scheduler.shutdown()


def main():
    parser = argparse.ArgumentParser(description='Agent Trader')
    parser.add_argument('--config', '-c', default='config.yaml', help='Config file path')
    parser.add_argument('--once', '-1', action='store_true', help='Run once and exit')
    parser.add_argument('--scheduled', '-s', action='store_true', help='Run scheduled (default)')
    
    args = parser.parse_args()
    
    agent = Agent(args.config)
    
    if args.once:
        result = agent.run_once()
        print(f"\nResult: {result}")
    else:
        agent.start_scheduled()


if __name__ == "__main__":
    main()
