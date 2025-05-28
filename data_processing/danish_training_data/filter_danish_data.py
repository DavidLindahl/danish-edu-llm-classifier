import pandas as pd
import re
import os
from typing import Tuple
import logging

class SimpleContentFilter:
    def __init__(self):
        self.setup_logging()
        self.setup_badwords()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def setup_badwords(self):
        """Setup Danish bad words list"""
        # Danish adult/sexual content
        self.bad_words = [
            'escort', 'prostitueret', 'luder', 'bordel', 'massage', 'tantra',
            'sex', 'porno', 'bryster', 'fisse', 'pik', 'kusse', 'anal',
            'blowjob', 'orgasme', 'swingerklub', 'stripper', 'nøgen', 'erotisk',
            'kneppe', 'sutter', 'sprøjte', 'sæd', 'dildo', 'webcam',
            'liderlig', 'frække', 'intim', 'yoni', 'lingam', 'milf', 'shemale',
            'gangbang', 'trekant', 'bdsm', 'fetish', 'latex', 'dominatrix',
            'callgirl', 'sexarbejder', 'pornostjerne', 'swinger', 'cam',
            'xxxs', 'hardcore', 'fetisch', 'kink', 'bondage', 'sadomaso',
            
            # Gambling
            'casino', 'poker', 'gambling', 'spillemaskiner', 'jackpot',
            'betting', 'væddemål', 'odds',
            
            # Financial spam
            'kviklån', 'forbrugslån', 'lånebeløb', 'rente', 'åop'
        ]
        
        # Convert to lowercase for matching
        self.bad_words = [word.lower() for word in self.bad_words]
    
    def count_bad_words(self, text: str) -> int:
        """Count bad words in text"""
        text_lower = text.lower()
        count = 0
        
        for bad_word in self.bad_words:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(bad_word) + r'\b'
            matches = len(re.findall(pattern, text_lower))
            count += matches
            
        return count
    
    def is_problematic(self, text: str, threshold: int = 2) -> Tuple[bool, str]:
        """
        Check if text is problematic based on bad word count
        
        Args:
            text: Text to check
            threshold: Number of bad words to trigger filtering
        """
        if not text or len(text.strip()) < 10:
            return True, "Text too short"
        
        bad_word_count = self.count_bad_words(text)
        
        # Check for spam patterns (lots of repeated characters/words)
        if self.is_spam_like(text):
            return True, "Spam-like content"
        
        # Flag if too many bad words
        if bad_word_count >= threshold:
            return True, f"Contains {bad_word_count} inappropriate words"
        
        return False, "Clean content"
    
    def is_spam_like(self, text: str) -> bool:
        """Simple spam detection"""
        # Check for table-like structures (common in your spam data)
        if text.count('||') > 5 or text.count('|') > 15:
            return True
        
        # Check for excessive punctuation
        punct_count = len(re.findall(r'[^\w\s]', text))
        if len(text) > 0 and punct_count / len(text) > 0.3:
            return True
        
        # Check for very repetitive content
        words = text.split()
        if len(words) > 10:
            unique_words = len(set(words))
            if unique_words / len(words) < 0.3:  # Less than 30% unique words
                return True
        
        return False
    
    def filter_dataset(self, data: pd.DataFrame, text_column: str = 'text', 
                      bad_word_threshold: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filter dataset into clean and problematic content
        
        Args:
            data: DataFrame to filter
            text_column: Column containing text to filter
            bad_word_threshold: Number of bad words to trigger filtering
        """
        df = data.copy()
        
        # Apply filtering
        filter_results = []
        bad_word_counts = []
        
        for text in df[text_column]:
            is_prob, reason = self.is_problematic(str(text), bad_word_threshold)
            bad_count = self.count_bad_words(str(text))
            
            filter_results.append((is_prob, reason))
            bad_word_counts.append(bad_count)
        
        # Add results to dataframe
        df['is_problematic'] = [r[0] for r in filter_results]
        df['filter_reason'] = [r[1] for r in filter_results]
        df['bad_word_count'] = bad_word_counts
        
        # Split data
        clean_data = df[~df['is_problematic']].drop([
            'is_problematic', 'filter_reason', 'bad_word_count'
        ], axis=1)
        
        problematic_data = df[df['is_problematic']]
        
        self.logger.info(f"Filtered {len(df)} samples: {len(clean_data)} clean, {len(problematic_data)} problematic")
        
        return clean_data, problematic_data
    
    def get_stats(self, problematic_data: pd.DataFrame) -> dict:
        """Get filtering statistics"""
        if len(problematic_data) == 0:
            return {"total_filtered": 0}
        
        return {
            "total_filtered": len(problematic_data),
            "avg_bad_words": problematic_data['bad_word_count'].mean(),
            "max_bad_words": problematic_data['bad_word_count'].max(),
            "filter_reasons": problematic_data['filter_reason'].value_counts().to_dict()
        }

def main():
    """Filter content from CSV file"""
    
    # Initialize filter
    content_filter = SimpleContentFilter()
    
    # Load data

    csv_path = os.path.join("data", "danish_unfiltered_data.csv")

    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found!")
        return
    
    df = pd.read_csv(csv_path, encoding="utf-8")
    print(f"Loaded {len(df)} samples")
    
    # Filter content (adjust threshold as needed: 1=strict, 3=lenient)
    clean_data, problematic_data = content_filter.filter_dataset(df, 
                                                               text_column='text', 
                                                               bad_word_threshold=2)
    
    # Save results
    os.makedirs("data", exist_ok=True)
    clean_data.to_csv("data/danish_filtered_data.csv", index=False, encoding="utf-8")
    # problematic_data.to_csv("data/problematic_data.csv", index=False, encoding="utf-8")
    
    # Print stats
    stats = content_filter.get_stats(problematic_data)
    
    print(f"\n=== Filtering Results ===")
    print(f"Original: {len(df)} samples")
    print(f"Clean: {len(clean_data)} samples ({len(clean_data)/len(df)*100:.1f}%)")
    print(f"Filtered: {len(problematic_data)} samples ({len(problematic_data)/len(df)*100:.1f}%)")
    
    if len(problematic_data) > 0:
        print(f"\nAverage bad words per filtered sample: {stats['avg_bad_words']:.1f}")
        print(f"Max bad words in a sample: {stats['max_bad_words']}")
        print(f"\nFilter reasons:")
        for reason, count in stats['filter_reasons'].items():
            print(f"  {reason}: {count}")
    
    print(f"\nFiles saved:")
    print(f"  Clean data: data/clean_data.csv")
    print(f"  Problematic data: data/problematic_data.csv")

if __name__ == "__main__":
    main()