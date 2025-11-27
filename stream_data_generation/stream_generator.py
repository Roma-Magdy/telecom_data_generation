# Fixed stream_generator.py
"""
Stream generator with complete real-time streaming methods
"""
import json
import time
import threading
import random
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from kafka import KafkaProducer
from kafka.errors import KafkaError
import logging
import glob
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
TAX_RATE = 0.14
TOPUP_CHANNELS = ['online', 'offline', 'hybrid']
TOPUP_WEIGHTS = [0.41, 0.41, 0.18]

class TelecomStreamGenerator:
    def __init__(self, bootstrap_servers='localhost:9092', batch_dir='../data_generation/output_batch'):
        """Initialize with batch data awareness"""
        self.batch_dir = batch_dir
        
        # Initialize Kafka Producer with proper configuration
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
            key_serializer=lambda k: str(k).encode('utf-8') if k else None,
            acks=1,
            retries=3,
            batch_size=16384,
            linger_ms=100,
            buffer_memory=33554432,
            max_block_ms=5000  # Don't block forever
        )
        
        # Load all data
        self.customers = self._load_all_chunks('customers')
        self.subscriptions = self._load_all_chunks('subscriptions')
        
        # Separate active and churned
        self.active_subs = self.subscriptions[self.subscriptions['subscription_end_date'].isna()]
        self.churned_subs = self.subscriptions[self.subscriptions['subscription_end_date'].notna()]
        
        # Separate by type for topups
        self.prepaid_subs = self.active_subs[self.active_subs['subscription_type'] == 'prepaid']
        self.prepaid_churned = self.churned_subs[self.churned_subs['subscription_type'] == 'prepaid']
        
        logger.info(f"Loaded {len(self.customers)} customers")
        logger.info(f"Active subscriptions: {len(self.active_subs)}")
        logger.info(f"Churned subscriptions: {len(self.churned_subs)}")
        logger.info(f"Prepaid active: {len(self.prepaid_subs)}")
    
    def _load_all_chunks(self, table_name):
        """Load and combine all chunks for a table"""
        pattern = f"{self.batch_dir}/{table_name}_chunk*.parquet"
        files = glob.glob(pattern)
        
        if not files:
            single_file = f"{self.batch_dir}/{table_name}.parquet"
            if os.path.exists(single_file):
                return pd.read_parquet(single_file)
            else:
                logger.error(f"No files found for {table_name}")
                return pd.DataFrame()
        
        dfs = []
        for file in sorted(files):
            dfs.append(pd.read_parquet(file))
        
        return pd.concat(dfs, ignore_index=True)
    
    def generate_historical_data(self):
        """Generate historical data for churned customers with proper flow control"""
        logger.info("Generating historical data for churned customers...")
        
        historical_count = 0
        batch_size = 0
        
        for idx, (_, sub) in enumerate(self.churned_subs.iterrows()):
            start_date = pd.to_datetime(sub['subscription_start_date'])
            end_date = pd.to_datetime(sub['subscription_end_date'])
            active_days = (end_date - start_date).days
            
            if active_days <= 0:
                continue
            
            # Generate CDRs (4 per month average)
            num_cdrs = min(int((active_days / 30) * 4), 100)
            for _ in range(num_cdrs):
                event_time = start_date + timedelta(
                    days=random.randint(0, max(1, active_days)),
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59)
                )
                
                cdr_event = {
                    'cdr_id': str(uuid.uuid4()),
                    'subscription_id': str(sub['subscription_id']),
                    'call_duration_seconds': max(1, int(np.random.lognormal(mean=2.0, sigma=1.0))),
                    'consumed_mb': round(float(np.random.lognormal(mean=1.5, sigma=1.2)), 3),
                    'timestamp': event_time.isoformat()
                    # Removed 'is_historical' field to maintain schema consistency
                }
                
                self.producer.send('cdr', 
                                 key=cdr_event['subscription_id'], 
                                 value=cdr_event)
                historical_count += 1
                batch_size += 1
                
                # Flush periodically
                if batch_size >= 100:
                    self.producer.flush(timeout=10)
                    batch_size = 0
                    time.sleep(0.01)
            
            # Generate topups for prepaid churned
            if sub['subscription_type'] == 'prepaid':
                num_topups = min(int((active_days / 30) * 5), 50)
                for _ in range(num_topups):
                    event_time = start_date + timedelta(
                        days=random.randint(0, max(1, active_days)),
                        hours=random.randint(0, 23)
                    )
                    
                    amount = round(float(np.random.lognormal(mean=3.0, sigma=0.5)), 2)
                    
                    topup_event = {
                        'topup_id': str(uuid.uuid4()),
                        'subscription_id': str(sub['subscription_id']),
                        'amount': amount,
                        'tax_amount': round(amount * TAX_RATE, 2),
                        'topup_channel': np.random.choice(TOPUP_CHANNELS, p=TOPUP_WEIGHTS),
                        'timestamp': event_time.isoformat()
                    }
                    
                    self.producer.send('topups',
                                     key=topup_event['subscription_id'],
                                     value=topup_event)
                    historical_count += 1
                    batch_size += 1
                    
                    if batch_size >= 100:
                        self.producer.flush(timeout=10)
                        batch_size = 0
                        time.sleep(0.01)
            
            # Generate support tickets (rare)
            num_tickets = min(int(np.random.poisson(0.15 * (active_days / 30))), 10)
            for _ in range(num_tickets):
                created_at = start_date + timedelta(
                    days=random.randint(0, max(1, active_days)),
                    hours=random.randint(0, 23)
                )
                
                # 80% resolved
                if random.random() < 0.80:
                    resolved_at = created_at + timedelta(days=random.randint(1, 30))
                    if resolved_at > end_date:
                        resolved_at = end_date
                    status = random.choice(['closed', 'closed', 'closed', 'escalated'])
                else:
                    resolved_at = None
                    status = 'open'
                
                ticket_event = {
                    'ticket_id': str(uuid.uuid4()),
                    'customer_id': str(sub['customer_id']),
                    'subscription_id': str(sub['subscription_id']),
                    'category': random.choice(['billing', 'network', 'device', 'sim']),
                    'created_at': created_at.isoformat(),
                    'resolved_at': resolved_at.isoformat() if resolved_at else None,
                    'status': status,
                    'agent_id': random.randint(1000, 2000),
                    'priority': np.random.choice(['low', 'medium', 'high'], p=[0.6, 0.3, 0.1])
                }
                
                self.producer.send('support_tickets',
                                 key=ticket_event['subscription_id'],
                                 value=ticket_event)
                historical_count += 1
                batch_size += 1
                
                if batch_size >= 100:
                    self.producer.flush(timeout=10)
                    batch_size = 0
                    time.sleep(0.01)
            
            # Progress update
            if historical_count % 1000 == 0:
                logger.info(f"Generated {historical_count} historical events...")
                self.producer.flush(timeout=10)
                time.sleep(0.5)
        
        # Final flush
        self.producer.flush(timeout=30)
        logger.info(f"✓ Historical data complete: {historical_count} events")
    
    def generate_cdr_stream(self):
        """Generate continuous CDR events for active subscriptions"""
        logger.info("Starting CDR stream generator...")
        
        while True:
            try:
                # Pick a random active subscription
                if len(self.active_subs) == 0:
                    time.sleep(5)
                    continue
                
                sub = self.active_subs.sample(1).iloc[0]
                
                # Generate CDR event
                cdr_event = {
                    'cdr_id': str(uuid.uuid4()),
                    'subscription_id': str(sub['subscription_id']),
                    'call_duration_seconds': max(1, int(np.random.lognormal(mean=2.0, sigma=1.0))),
                    'consumed_mb': round(float(np.random.lognormal(mean=1.5, sigma=1.2)), 3),
                    'timestamp': datetime.now().isoformat()
                }
                
                self.producer.send('cdr',
                                 key=cdr_event['subscription_id'],
                                 value=cdr_event)
                
                # Flush periodically
                if random.random() < 0.1:  # 10% chance
                    self.producer.flush(timeout=5)
                
                # Variable delay to simulate realistic traffic
                delay = np.random.exponential(0.5)  # Average 0.5 seconds between events
                time.sleep(min(delay, 5))  # Cap at 5 seconds
                
            except Exception as e:
                logger.error(f"Error in CDR stream: {e}")
                time.sleep(1)
    
    def generate_topups_stream(self):
        """Generate continuous topup events for prepaid subscriptions"""
        logger.info("Starting topups stream generator...")
        
        while True:
            try:
                # Only prepaid subscriptions have topups
                if len(self.prepaid_subs) == 0:
                    time.sleep(5)
                    continue
                
                sub = self.prepaid_subs.sample(1).iloc[0]
                
                # Generate topup event
                amount = round(float(np.random.lognormal(mean=3.0, sigma=0.5)), 2)
                
                topup_event = {
                    'topup_id': str(uuid.uuid4()),
                    'subscription_id': str(sub['subscription_id']),
                    'amount': amount,
                    'tax_amount': round(amount * TAX_RATE, 2),
                    'topup_channel': np.random.choice(TOPUP_CHANNELS, p=TOPUP_WEIGHTS),
                    'timestamp': datetime.now().isoformat()
                }
                
                self.producer.send('topups',
                                 key=topup_event['subscription_id'],
                                 value=topup_event)
                
                # Flush periodically
                if random.random() < 0.1:
                    self.producer.flush(timeout=5)
                
                # Topups are less frequent than CDRs
                delay = np.random.exponential(2.0)  # Average 2 seconds between events
                time.sleep(min(delay, 10))
                
            except Exception as e:
                logger.error(f"Error in topups stream: {e}")
                time.sleep(1)
    
    def generate_support_tickets_stream(self):
        """Generate continuous support ticket events"""
        logger.info("Starting support tickets stream generator...")
        
        while True:
            try:
                if len(self.active_subs) == 0:
                    time.sleep(5)
                    continue
                
                sub = self.active_subs.sample(1).iloc[0]
                
                # Support tickets are rare
                if random.random() > 0.01:  # Only 1% chance
                    time.sleep(10)
                    continue
                
                # Generate ticket event
                created_at = datetime.now()
                
                # 80% get resolved
                if random.random() < 0.80:
                    resolved_at = created_at + timedelta(days=random.randint(1, 30))
                    status = random.choice(['closed', 'closed', 'closed', 'escalated'])
                else:
                    resolved_at = None
                    status = 'open'
                
                ticket_event = {
                    'ticket_id': str(uuid.uuid4()),
                    'customer_id': str(sub['customer_id']),
                    'subscription_id': str(sub['subscription_id']),
                    'category': random.choice(['billing', 'network', 'device', 'sim']),
                    'created_at': created_at.isoformat(),
                    'resolved_at': resolved_at.isoformat() if resolved_at else None,
                    'status': status,
                    'agent_id': random.randint(1000, 2000),
                    'priority': np.random.choice(['low', 'medium', 'high'], p=[0.6, 0.3, 0.1])
                }
                
                self.producer.send('support_tickets',
                                 key=ticket_event['subscription_id'],
                                 value=ticket_event)
                
                # Flush
                if random.random() < 0.2:
                    self.producer.flush(timeout=5)
                
                # Support tickets are very rare
                delay = np.random.exponential(30)  # Average 30 seconds
                time.sleep(min(delay, 60))
                
            except Exception as e:
                logger.error(f"Error in support tickets stream: {e}")
                time.sleep(1)
    
    def start(self, generate_historical=True):
        """Start generators with optional historical data"""
        
        # if generate_historical and len(self.churned_subs) > 0:
        #     logger.info("First, generating historical data for churned customers...")
        #     self.generate_historical_data()
        #     logger.info("Historical data complete. Starting real-time streams...")
        
        # Now start real-time streams for active customers
        threads = [
            threading.Thread(target=self.generate_cdr_stream, daemon=True),
            threading.Thread(target=self.generate_topups_stream, daemon=True),
            threading.Thread(target=self.generate_support_tickets_stream, daemon=True)
        ]
        
        for thread in threads:
            thread.start()
            
        logger.info("All stream generators started. Press Ctrl+C to stop...")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            self.producer.close()

if __name__ == "__main__":
    generator = TelecomStreamGenerator(bootstrap_servers='localhost:9092')
    generator.start(generate_historical=True)