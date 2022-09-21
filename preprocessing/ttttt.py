class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        # 在j-i+1>2时
        # if p[i+1][j-1]==True and s[i]==s[j]  
        #   p[i][j] = True:
        # else:
        #   p[i][j]=False  
        if len(s)==0:
            return ""
        begin_idx,end_idx = 0,0
        now_len = -1
        # p = [[0]*len(s)]*len(s)
        p = [[0] * len(s) for i in range(len(s))]
        for str_len in range(1,len(s)+1):
            for first_point in range(len(s)):
                last_point = first_point+str_len-1
                if last_point<len(s):
                    if str_len == 1:
                        p[first_point][last_point]=True
                    elif str_len == 2:
                        if s[last_point]==s[first_point]:
                            p[first_point][last_point]=True
                        else:
                            print(s[first_point:last_point+1],first_point,last_point)
                            p[first_point][last_point]=False
                    else:
                        if s[last_point]==s[first_point] and p[first_point+1][last_point-1]:
                            p[first_point][last_point]=True
                        else:
                            p[first_point][last_point]=False
                    if p[first_point][last_point]:
                        now_len = last_point-first_point+1
                        begin_idx = first_point
                        end_idx = last_point
                    print(first_point,last_point,p[first_point][last_point],begin_idx,end_idx)
        return s[begin_idx:end_idx+1]

s = Solution()
print(s.longestPalindrome("babad"))
