import pandas as pd
import json

def convert_dataframe_to_messages(df):
    """
    Convert DataFrame with columns ["system", "user", "assistant"] to messages format
    
    Args:
        df: pandas DataFrame with columns ["system", "user", "assistant"]
        
    Returns:
        List of dictionaries with "messages" key containing conversation format
    """
    results = []
    
    for idx, row in df.iterrows():
        messages = []
        
        # Add system message if present and not null
        if pd.notna(row["system"]) and row["system"].strip():
            messages.append({
                "role": "system", 
                "content": row["system"].strip()
            })
        
        # Add user message if present and not null
        if pd.notna(row["user"]) and row["user"].strip():
            messages.append({
                "role": "user", 
                "content": row["user"].strip()
            })
        
        # Add assistant message if present and not null
        if pd.notna(row["assistant"]) and row["assistant"].strip():
            messages.append({
                "role": "assistant", 
                "content": row["assistant"].strip()
            })
        
        # Only add if we have at least one message
        if messages:
            results.append({"messages": messages})
    
    return results

def convert_dataframe_to_messages_multiturn(df):
    """
    Convert DataFrame with user_1, assistant_1, user_2, assistant_2, ..., system columns
    to messages format for MultiTurnSFTDataset
    
    Args:
        df: DataFrame with columns like user_1, assistant_1, user_2, assistant_2, system
        
    Returns:
        List of dictionaries with "messages" key
    """
    results = []
    
    for idx, row in df.iterrows():
        messages = []
        
        # Add system message first if present
        if 'system' in row and pd.notna(row['system']) and str(row['system']).strip():
            messages.append({
                "role": "system",
                "content": str(row['system']).strip()
            })
        
        # Find all user and assistant columns
        user_cols = [col for col in df.columns if col.startswith('user_')]
        assistant_cols = [col for col in df.columns if col.startswith('assistant_')]
        
        # Sort by number to ensure correct order
        user_cols.sort(key=lambda x: int(x.split('_')[1]))
        assistant_cols.sort(key=lambda x: int(x.split('_')[1]))
        
        # Get the maximum turn number
        max_user_turn = len(user_cols)
        max_assistant_turn = len(assistant_cols)
        max_turns = max(max_user_turn, max_assistant_turn)
        
        # Interleave user and assistant messages
        for turn in range(1, max_turns + 1):
            user_col = f"user_{turn}"
            assistant_col = f"assistant_{turn}"
            
            # Add user message if present
            if user_col in row and pd.notna(row[user_col]) and str(row[user_col]).strip():
                messages.append({
                    "role": "user",
                    "content": str(row[user_col]).strip()
                })
            
            # Add assistant message if present  
            if assistant_col in row and pd.notna(row[assistant_col]) and str(row[assistant_col]).strip():
                messages.append({
                    "role": "assistant", 
                    "content": str(row[assistant_col]).strip()
                })
        
        # Only add if we have at least one message
        if messages:
            results.append({"messages": messages})
    
    return results
