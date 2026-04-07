def edit_distance(s1, s2):
    """
    Compute the minimum edit distance (Levenshtein) between two strings.
    Returns: int - the minimum number of operations.
    """
    m = len(s1)
    n = len(s2)

    # 1. Create a (m+1) x (n+1) DP table
    # dp[i][j] will store the distance between s1[0...i-1] and s2[0...j-1]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 2. Base cases: Transforming a string into an empty string
    # Distance from s1[0...i] to "" is i (i deletions)
    for i in range(m + 1):
        dp[i][0] = i
    # Distance from "" to s2[0...j] is j (j insertions)
    for j in range(n + 1):
        dp[0][j] = j

    # 3. Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Check if characters at the current positions match
            if s1[i - 1] == s2[j - 1]:
                # Characters match: No operation needed, take diagonal value
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Characters differ: Take 1 + min of three possible operations
                # dp[i-1][j]   -> Deletion
                # dp[i][j-1]   -> Insertion
                # dp[i-1][j-1] -> Substitution (Replacement)
                dp[i][j] = 1 + min(dp[i - 1][j],      
                                   dp[i][j - 1],      
                                   dp[i - 1][j - 1])  

    # The result is the value in the bottom-right cell
    return dp[m][n]