'''
1.确定状态
2.确定dp函数的定义
3.确定选择并择优
4.明确base case

最优子结构
1.遍历的过程，所需的状态必须是已经计算出来的
2.遍历的终点必须是存储结果的那个位置

'''


# def cmp(a,b):
#  # 如果返回的是一个大于0的值，那么代表a>b
#  # 如果返回的是一个小于0的值，那么代表a<b
#  # 如果返回的是一个等于0的值，那么代表a=b
from functools import cmp_to_key
def cmp_new(x, y):
    if x[0] > y[0]:
        return 1
    elif x[0] < y[0]:
        return -1
    else:
        return 0
t = [[1,2], [0, 3], [4,1],[2,5]]
sorted(t, key=cmp_to_key(cmp_new))


class Solution(object):
    """"""
    def coinCharge(self, coins, amount):
        dp = [amount+1 for _ in range(amount+1)]
        dp[0] = 0
        for i in range(len(dp)):
            for coin in coins:
                if i - coin < 0:
                    continue
                dp[i] = min(dp[i], 1+dp[i-coin])
        return dp[amount] if dp[amount] != amount+1 else -1

    def lengthOfLIS(self, nums):
        dp = [1 for _ in range(len(nums))]
        for i in range(len(nums)):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j]+1)

        res = max(dp)
        return res

    def minEditDistance(self, s1, s2):
        m = len(s1)
        n = len(s2)
        dp = [[0 for _ in range(n+1)] for _ in  range(m+1)]
        #base case
        for i in range(1, m+1):
            dp[i][0] = i
        for j in range(1, n+1):
            dp[0][j] = j
        # 自底向上求解
        for i in range(1, m+1):
            for j in range(1, n+1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(
                        dp[i-1][j] + 1, # 删除
                        dp[i][j-1] + 1, # 插入
                        dp[i-1][j-1] + 1 # 替换或路过
                    )
        return dp[m][n]

    def superEggDrop(self, K, N):
        memo = {}
        '''
        dp(K,N) = min_0<=i<=N{max(dp(K-1, i-1), dp(K, N-i))+1} 
        '''
        def dp(K, N):
            # base case
            if K == 1: return N
            if N == 0: return 0
            # 避免重复计算
            if (K, N) in memo:
                return memo[(K, N)]
            res = float('INF')
            # 穷举所有可能的选择
            for i in range(1, N+1):
                res = min(res,
                          max(
                              dp(K, N-i),
                              # 没碎 [1..N] => [i+1...N] 共N-i层
                              dp(K-1, i-1)
                              # 碎 [1..N] => [1...i-1] 共i-1层
                          ) + 1 # 在i楼扔了一次
                        )
                # 记入备忘录
                memo[(K, N)] = res
            return res
        return dp(K, N)

    """
    dp 数组的定义，确定当前的鸡蛋个数和最多允许的 扔鸡蛋次数，就知道能够确定 F 的最⾼楼层数。
    while 循环结束的条件是 dp[K][m] == N ，也就是给你 K 个鸡蛋， 测试 m 次，最坏情况下最多能测试 N 层楼
    1、⽆论你在哪层楼扔鸡蛋，鸡蛋只可能摔碎或者没摔碎，碎了的话就测楼 下，没碎的话就测楼上。 
    2、⽆论你上楼还是下楼，总的楼层数 = 楼上的楼层数 + 楼下的楼层数 + 1（当前这层楼）。 
    根据这个特点，可以写出下⾯的状态转移⽅程： 
        dp[k][m] = dp[k][m - 1] + dp[k - 1][m - 1] + 1 
        dp[k][m - 1] 就是楼上的楼层数，因为鸡蛋个数 k 不变，也就是鸡蛋没 碎，扔鸡蛋次数 m 减⼀； 
        dp[k - 1][m - 1] 就是楼下的楼层数，
        因为鸡蛋个数 k 减⼀，也就是鸡 蛋碎了，同时扔鸡蛋次数 m 减⼀。
    """
    def superEggDrop2(self, K, N):
        dp = [[0 for _ in range(K+1)] for _ in range(N+1)]
        m = 0
        while dp[K][m] < N:
            m += 1
            for k in range(1, K+1):
                dp[k][m] = dp[k][m-1] + dp[k-1][m-1] + 1
        return m

    def longesPalindromeSubseq(self, s):
        """
        s[i] 和 s[j] 的字符
        如果它俩相等，那么它俩加上 s[i+1..j-1] 中的最⻓回⽂⼦序列就是 s[i..j] 的最⻓回⽂⼦序列
        如果它俩不相等，说明它俩不可能同时出现在 s[i..j] 的最⻓回⽂⼦序列 中，那么把它俩分别加⼊ s[i+1..j-1] 中，看看哪个⼦串产⽣的回⽂⼦序 列更⻓即可
        if (s[i] == s[j]) // 它俩⼀定在最⻓回⽂⼦序列中
            dp[i][j] = dp[i + 1][j - 1] + 2;
        else// s[i+1..j] 和 s[i..j-1] 谁的回⽂⼦序列更⻓？
            dp[i][j] = max(dp[i + 1][j], dp[i][j - 1]);
        :param s:
        :return:
        """
        n = len(s)
        dp = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            dp[i][i] = 1
        # 反着遍历保证正确的状态转移
        for i in range(n-1, -1, -1):
            for j in range(i+1, n):
                # 状态转移方程
                if s[i] == s[j]:
                    dp[i][j] = dp[i+1][j-1] + 2
                else:
                    dp[i][j] = max(dp[i+1][j], dp[i][j-1])

        return dp[0][n-1]

    def stoneGame(self, piles):
        n = len(piles)
        # 初始化dp数组
        dp = [[[0, 0] for _ in range(n)] for _ in range(n)]
        for i in range(n):
            dp[i][i] = piles[i]
        #斜着遍历数组
        for l in range(2, n+1):
            for i in range(n-l):
                j = l+i-1
                # 先手选择最左边或最右边的分数
                left = piles[i] + dp[i+1][j][1]
                right = piles[i] + dp[i][j-1][1]
                # 套用状态转移方程
                if left > right:
                    dp[i][j][0] = left
                    dp[i][j][1] = dp[i+1][j][0]
                else:
                    dp[i][j][0] = right
                    dp[i][j][1] = dp[i][j-1][0]

        res = dp[0][n-1]
        return res[0] - res[1]

    def intervalSchedule(self, intvs):
        if len(intvs) == 0:
            return 0
        intvs = sorted(intvs, key=lambda x: x[1])
        # 至少有一个区间不相交
        count = 1
        #排序后，第一个区间就是x
        x_end = intvs[0][1]
        for inter in intvs:
            start = inter[0]
            if start >= x_end:
                count += 1
                x_end = inter[1]
        return count

    def KMP(self, pat):
        M = len(pat)
        # dp[状态][字符] = 下个状态
        dp = [[0 for _ in range(256)] for _ in range(M)]
        # base case
        dp[0][pat[0]] = 1
        # 影子状态 X 初始状态为0
        X = 0
        # 当前状态j 从1开始
        for j in range(1, M):
            for c in range(256):
                if pat[j] == c:
                    dp[j][c] = j+1
                else:
                    dp[j][c] = dp[X][c]

            # 更新影子状态
            X = dp[X][pat[j]]

        def search(text):
            N = len(text)
            # pat的初始状态为0
            j = 0
            for i in range(N):
                # 当前状态是j,遇到字符text[i]
                # 计算pat的下一个状态
                j = dp[j][text[i]]
                # 到达终止状态
                if j == M:
                    return i - M + 1
            return -1
        return search

    def maxProfit(self, prices):
        n = len(prices)
        dp = [[0, 0] for _ in range(n)]
        for i in range(n):
            if i - 1 == -1:
                dp[i][0] = 0
                dp[i][1] = -prices[i]
                continue
            dp[i][0] = max(dp[i-1][0], dp[i-1][1]+prices[i])
            dp[i][1] = max(dp[i-1][1], -prices[i])
        return dp[n-1][0]

    def maxProfit_with_k_inf(self, pricies):
        n = len(pricies)
        dp_i_0 = 0
        dp_i_1 = int('-inf')
        for i in range(n):
            temp = dp_i_0
            dp_i_0 = max(dp_i_0, dp_i_1 + pricies[i])
            dp_i_1 = max(dp_i_1, temp-pricies[i])
        return dp_i_1

    """
    每次 sell 之后要等⼀天才能继续交易。
    """
    def maxProfit_with_cool(self, prices):
        n = len(prices)
        dp_i_0 = 0
        dp_i_1 = int('-inf')
        # 代表dp[i-2][0]
        dp_pre_0 = 0
        for i in range(n):
            temp = dp_i_0
            dp_i_0 = max(dp_i_0, dp_i_1+prices[i])
            dp_i_1 = max(dp_i_1, dp_pre_0-prices[i])
            dp_pre_0 = temp
        return dp_i_0

    def maxProfix(self, prices, fee):
        n = len(prices)
        dp_i_0 = 0
        dp_i_1 = int('-inf')
        for i in range(n):
            temp = dp_i_0
            dp_i_0 = max(dp_i_0, dp_i_1+prices[i])
            dp_i_1 = max(dp_i_1, temp-prices[i]-fee)
        return dp_i_0

    def maxProfit_k_any(self, max_k, prices):
        n = len(prices)
        if max_k > n/2:
            return self.maxProfit_with_k_inf(prices)
        dp = [[[0, 0] for _ in range(max_k+1)] for _ in range(n)]
        for i in range(n):
            for k in range(max_k, 0, -1):
                if i - 1 == -1:
                    # base case
                    pass
                dp[i][k][0] = max(dp[i][k][0], dp[i-1][k][1]+prices[i])
                dp[i][k][1] = max(dp[i][k][1], dp[i-1][k-1][0]-prices[i])

        return dp[n-1][max_k][0]

    def rob(self, nums):
        n = len(nums)
        # dp[i] = x 表示从第i间房子抢劫，最多能抢劫的钱为x
        # base case dp[n] = 0
        dp = [0 for _ in range(n+2)]
        for i in range(n-1, -1, -1):
            dp[i] = max(dp[i+1], nums[i]+dp[i+2])
        return dp[0]

    def rob1(self, nums):
        n = len(nums)
        if n == 1: return nums[0]
        return max(self.robRange(nums, 0, n-2), self.robRange(nums, 1, n-1))

    def robRange(self, nums, start, end):
        n = len(nums)
        dp_i_1 = 0
        dp_i_2 = 0
        dp_i = 0
        for i in range(end, start-1, -1):
            dp_i = max(dp_i_1, nums[i] + dp_i_2)
            dp_i_2 = dp_i_1
            dp_i_1 = dp_i
        return dp_i

    def rob_tree(self, root):
        if not root:
            return [0, 0]
        '''
            arr = [0, 0]
            arr[0]表示不抢root的话， 得到最大的钱数
            arr[1]表示抢root的话，得么的最大的钱数
        '''
        left = self.rob_tree(root.left)
        right = self.rob_tree(root.right)
        _rob = root.val + left[0] + right[0]
        not_rob = max(left[0] + left[1], right[0] + right[1])
        return [not_rob, _rob]

    memo = {}
    def rob_tree2(self, root):
        if not root: return 0
        if root in self.memo:
            return self.memo[root]
        # 抢，然后去下下家
        do_it = root.val
        if root.left:
            do_it += self.rob_tree2(root.left.left) + self.rob_tree2(root.left.right)
        if root.right:
            do_it += self.rob_tree2(root.right.left) + self.rob_tree2(root.right.right)
        # 不抢，然后去下家
        not_do = self.rob_tree2(root.left) + self.rob_tree2(root.right)
        res = max(do_it, not_do)
        self.memo[root] = res
        return res

    def maxA(self, N):
        dp = [0 for _ in range(N+1)]
        dp[0] = 0
        for i in range(1, N+1):
            # A键
            dp[i] = dp[i-1] + 1
            for j in range(2, i):
                #  全选 & 复制 dp[j-2]，连续粘贴 i - j 次
                # 屏幕上共 dp[j - 2] * (i - j + 1) 个 A
                dp[i] = max(dp[i], dp[j-2]*(i-j+1))

        return dp[N]

    def isMatch(self, text, pattern):
        memo = {}
        def dp(i, j):
            if (i, j) in memo:
                return memo[(i, j)]
            if j == len(pattern):
                return i == len(text)
            first = i < len(text) and pattern[j] in {text[i], '.'}
            # 解释：如果发现有字符和 '*' 结合，
            # 或者匹配该字符 0 次，然后跳过该字符和 '*'
            # 或者当 pattern[0] 和 text[0] 匹配后，移动 text
            if j < len(pattern) - 2 and pattern[j+1] == '*':
                ans = dp(i, j+2) or first and dp(i+1, j)
            else:
                ans = first and dp(i+1, j+1)
            memo[(i, j)] = ans
            return ans
        return dp(0, 0)

    def longesCommonSubsequence(self, s1, s2) -> int:
        m, n = len(s1), len(s2)
        # 构建 DP table 和 base case
        dp = [[0] * (n+1) for _ in range(m+1)]
        # 进行状态转移
        for i in range(1, m+1):
            for j in range(1, n+1):
                if s1[i-1] == s2[j-1]:
                    # 找到一个Lcs中的字符
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1]

    def levelOrder(self, root):
        ret = []
        if not root:
            return ret
        q = []
        q.append(root)
        while q:
            size = len(q)
            ret.append([])
            for _ in range(size):
                node = q[0]
                q = q[1:]
                ret[-1].append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
        return ret

    # 递归反转链表
    def reverse(self, head):
        if not head.next:
            return head
        last = self.reverse(head.next)
        head.next.next = head
        head.next = None
        return last

    # 反转前n个
    # 后驱节点
    successor = None
    def reverseN(self, head, n):
        if n == 1:
            self.successor = head.next
            return head
        last = self.reverseN(head.next, n-1)
        head.next.next = head
        # 让反转后的head节点和后面的节点连接起来
        head.next = self.successor
        return last

    #反转区间的
    def reverseBetween(self, head, m, n):
        if n == 1:
            return self.reverseN(head, n)
        #前进到反转的起点触发base case
        head.next = self.reverseBetween(head.next, m-1, n-1)
        return head

    def hasCycle(self, head):
        fast = head
        slow = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                return True
        return False

    def detectCycle(self, head):
        fast = head
        slow = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                break
        slow = head
        while slow != fast:
            fast = fast.next
            slow = slow.next
        return slow

    def searchListK(self, head, k):
        slow = head
        fast = head
        for i in range(k):
            fast = fast.next
        while fast:
            slow = slow.next
            fast = fast.next
        return slow

    def twoSum(self, nums, target):
        left = 0
        right = len(nums) - 1
        while left < right:
            _sum = nums[left] + nums[right]
            if _sum == target:
                return [left+1, right+1]
            elif _sum < target:
                left += 1
            elif _sum > target:
                right -= 1
        return [-1, -1]

    def reverse_array(self, nums):
        left = 0
        right = len(nums) - 1
        while left < right:
            temp = nums[left]
            nums[left] = nums[right]
            nums[right] = nums[left]
            left += 1
            right -= 1
    def minWindow(self, s, t):
        start = 0
        minLen = int('inf')
        left = 0
        right = 0
        window = {}
        needs = {}
        for c in t:
            needs[c] = needs.get(c, 0)+1
        match = 0
        while right < len(s):
            c1 = s[right]
            if c1 in needs:
                window[c1] = window.get(c1, 0)+1
                if window[c1] == needs[c1]:
                    match +=1
            right += 1
            while match == len(needs):
                if right - left < minLen:
                    start = left
                    minLen = right-left
                c2 = s[left]
                if c2 in needs:
                    window[c2] = window.get(c2,0)-1
                    if window[c2] < needs[c2]:
                        match -= 1
                left +=1

        return ''  if minLen == int('int') else  s[start, start+minLen]

    def findAnagrams(self, s, t):
        start = 0
        # minLen = int('inf')
        res = []
        left = 0
        right = 0
        window = {}
        needs = {}
        for c in t:
            needs[c] = needs.get(c, 0)+1
        match = 0
        while right < len(s):
            c1 = s[right]
            if c1 in needs:
                window[c1] = window.get(c1, 0)+1
                if window[c1] == needs[c1]:
                    match +=1
            right += 1
            while match == len(needs):
                if right - left == len(t):
                    res.append(left)
                c2 = s[left]
                if c2 in needs:
                    window[c2] = window.get(c2,0)-1
                    if window[c2] < needs[c2]:
                        match -= 1
                left +=1

        return ''  if minLen == int('int') else  s[start, start+minLen]

    def lengthOfLongestSubstring(self, s):
        left = 0
        right = 0
        window = {}
        res = 0
        while right < len(s):
            c1 = s[right]
            window[c1] = window.get(c1, 0) + 1
            right += 1
            # 如果window中出现重复字符
            # 开始移动left缩小窗口
            while window.get(c1, 0) > 1:
                c2 = s[left]
                window[c2] = window.get(c2, 0)-1
                left -= 1
            res = max(res, right-left)
        return res

    def twoSum_(self, nums, target):
        n = len(nums)
        index = {}
        for i in range(n):
            index[nums[i]] = i
        for i in range(n):
            other = target-nums[i]
            # 如果other存在，且不是nums[i]本身
            if other in index and index[other] != i:
                return [i, index[other]]
        return [-1, -1]

    def hammingWeight(self, n):
        res = 0
        while n:
            n = n&(n-1)
            res += 1
        return res

    def isPowerOfTwo(self, n):
        if n <= 0:
            return False
        return n&(n-1) == 0

    def subarraySum(self, nums, k):
        n = len(nums)
        #前缀和， 该前缀和出现的次数
        preSum = {}
        # base case
        preSum[0] = 1
        ans = 0
        sum0_i = 0
        for i in range(n):
            sum0_i += nums[i]
            #这是我们要找的前缀和 nums[0..j]
            sum0_j = sum0_i-k
            #如果前面有这个前缀和，则直接更新答案
            if sum0_j in preSum:
                ans += preSum.get(sum0_j, 0)
            # 把前缀和 sums[0..j]加入并记录出现次数
            preSum[sum0_i] = preSum.get(sum0_i, 0)+1
        return ans

    def multiply(self, num1, num2):
        m = len(num1)
        n = len(num2)
        res = [0 for _ in range(m+n)]
        for i in range(m-1, -1, -1):
            for j in range(n-1, -1, -1):
                mul = int(num1[i]) * int(num2[j])
                # 乘积在res对应的位置
                p1 = i+j
                p2 = i+j+1
                # 叠加到res上
                _sum = mul+res[p2]
                res[p2] = _sum % 10
                res[p1] += _sum/10
        #结果前可能存在0
        i = 0
        while i < len(res) and res[i] == 0:
            i += 1
        # 将计算结果字符串化
        return ''.join(res[i:])

    def mergeIntervals(self, intervals):
        if not intervals:
            return []
        # 按区间的start升序排列
        intervals = sorted(intervals, key=lambda x: x[0])
        res = []
        res.append(intervals[0])
        for i in range(1, len(intervals)):
            curr = intervals[i]
            #res 中最后一个引用
            last = res[-1]
            if curr[0] < last[1]:
                last[1] = max(last[1], curr[1])
            else:
                # 处理下一个告诉合并区间
                res.append(curr)
        return res

    def intervalIntersection(self, A, B):
        i = 0
        j = 0
        res = []
        while i < len(A) and j < len(B):
            a1, a2 = A[i][0], A[i][1]
            b1, b2 = B[i][0], B[i][1]
            # 两个区间存在交集
            if b2 >= a1 and a2 >= b1:
                #计算出交集
                res.append([max(a1, b1), min(a2, b2)])
            # 指针前进
            if b2 < a2:
                j += 1
            else:
                i += 1
        return res


    ## 二叉树节点和为value的路径数
    def pathSum(self, root, val):
        if not root:
            return 0
        # 以自己为开头的路径数
        pathImLeading = self.pathSumCount(root, val)
        # 左边中径总数
        leftPathSum = self.pathSum(root.left, val)
        # 右边路径总数
        rightPathSum = self.pathSum(root.right, val)
        return pathImLeading + leftPathSum + rightPathSum

    def pathSumCount(self, node, val):
        if not node:
            return 0
        # 我能不能独当一面，作为一条单独的路径呢
        isMe = 1 if node.val == val else 0
        # 左边的小老弟，你那边能凑向个val - node.val呀
        leftBrother = self.pathSumCount(node.left, val-node.val)
        # 右边的小老弟，你那边能凑几个val - node.val呀
        rightBrother = self.pathSumCount(node.right, val-node.val)
        return isMe + leftBrother + rightBrother

    def removeDuplicates(self, nums):
        n = len(nums)
        if n == 0:
            return 0
        slow = 0
        fast = 1
        while fast < n:
            if nums[fast] != nums[slow]:
                slow += 1
                # 维护nums[0..slow] 无重复
            fast += 1
        return slow+1

    def removeDuplicateList(self, head):
        if not head:
            return None
        slow = head
        fast = head.next
        while fast:
            if fast.val != slow.val:
                slow.next = fast
                slow = slow.next
            fast = fast.next
        slow.next = None
        return head

    def longestPalindrome(self, s):
        res = ''
        def palindrome(s, l, r):
            # 防止索引越界
            while l >= 0 and r < len(s) and s[l] == s[r]:
                l -= 1
                r += 1
            return s[l+1:r]
        for i in range(len(s)):
            # 以s[i]为中心的最长回文了串
            s1 = palindrome(s, i, i)
            # 以s[i]和s[i+1]为中心的最长文了串
            s2 = palindrome(s, i, i+1)
            res = res if len(res) > len(s1) else s1
            res = res if len(res) > len(s2) else s2
        return res

    def reverseList(self, head):
        pre = None
        cur = head
        nxt = head
        while cur:
            nxt = cur.next
            # 逐个结点反转
            cur.next = pre
            #更新指针位置
            pre = cur
            cur = nxt
        return pre

    #/** 反转区间 [a, b) 的元素，注意是左闭右开 */
    def reversRangeList(self, a, b):
        pre = None
        cur = a
        nxt = a
        while cur != b:
            nxt = cur.next
            cur.next = pre
            pre = cur
            cur = nxt
        return pre

    def reverseKGroup(self, head, k):
        if not head:
            return None
        # 区间[a,b)包含k个待反转元素
        a = head
        b = head
        for i in range(k):
            # 不足k个，不需要反转， base case
            if not b:
                return head
            b = b.next

        # 反转前k个元素
        new_head = self.reversRangeList(a, b)
        # 递归反转后续链表并连接起来
        a.next = self.reverseKGroup(b, k)
        return new_head

    def isBracketsValid(self, s):

        def lefeOf(c):
            if c == ')':
                return '('
            if c == '}':
                return '{'
            return '['

        left = []
        for c in s:
            if c in ['(', '[', '{']:
                left.append(c)
            else:
                # 字符c是右括号
                if left and lefeOf(c) == left[-1]:
                    left.pop()
                else:
                    # 和最近的左括号不匹配
                    return False
        return len(left) == 0

    def missingNumber(self, nums):
        # 只要把所有的元素和索引做异或运算，成对⼉的 数字都会消为 0，只有这个落单的元素会剩下
        n = len(nums)
        res = 0
        # 先和新补的索引异或一下
        res ^= n
        # 和其他的元素，索引做异或
        for i in range(n):
            res ^= i^nums[i]
        return res

    def missingNumbers(self, nums):
        n = len(nums)
        res = 0
        # 新补的索引
        res += n-0
        # 剩下索引和元素差加起来
        for i in range(n):
            res += i - nums[i]

        return res

    def isPalindrome2(self, s):
        left = 0
        right = len(s)-1
        while left < right:
            if s[left] != s[right]:
                return False
            left += 1
            right -= 1
        return True

    def isPalindromeLink(self, head):

        def reverse(head):
            pre = None
            cur = head
            while cur:
                nxt = cur.next
                cur.next = pre
                pre = cur
                cur = nxt
            return pre
        slow = head
        fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        # slow 指针现在指向链表中点
        # 如果 fast 指针没有指向 null ，说明链表⻓度为奇数， slow 还要再前 进⼀步
        if fast:
            slow = slow.next
        left = head
        right = reverse(slow)
        while right:
            if left.val != right.val:
                return False
            left = left.next
            right = right.next
        return True

    def isSubsequence(self, s, t):
        i = 0
        j = 0
        while i < len(s) and j < len(t):
            if s[i] == t[j]:
                j+=1
        return i == len(s)

    def countPrimes(self, n):
        '''
        O(N*loglogN)
        :param n:
        :return:
        '''
        isPrim = [1 for _ in range(n)]
        i = 2
        while i*i < n:
            if isPrim[i]:
                j = i*i
                while j < n:
                    isPrim[j] = 0
                    j += i
            i += 1
        count = 0
        for i in range(2, n):
            if isPrim[i]:
                count += 1
        return count

    def trainWater(self, heights):
        if not heights:
            return 0
        n = len(heights)
        left = 0
        right = n-1
        ans = 0
        l_max = heights[0]
        r_max = heights[n-1]
        while left < right:
            l_max = max(l_max, heights[left])
            r_max = max(r_max, heights[right])
            if l_max < r_max:
                ans += l_max - heights[left]
                left += 1
            else:
                ans += r_max - heights[right]
                right -= 1
        return ans















'''
滑动窗口
left = 0
right = 0
while right < len(s):
    window.add(s[right])
    right += 1
    while valid:
        window.remove(s[left])
        left += 1
'''

'''
1.判断两个数是否异号
x = -1
y = 2
=> x ^ y < 0

2.交换两个数
a = 1
b = 2
a ^= b
b ^= a
a ^= b

3.加一
n = 1
n = -~n

4.减一
n = 1
n = ~-n

5.消除n的二进制表示中的最后一个1
n&(n-1)

5.计算汉明权重(Hamming Weight)

6.一个数如果是2的指数，它的二进制一定只包含一个1

7.^异或运算：
    一个数和它本身做异或运算结果为0，一个数和0做异或运算还是它本身

分治算法：典型的有归并排序
分治算法三步走：分解－>解决->合并
'''


class BST(object):
    def isValidBST(self, root, _min, _max):
        if not root:
            return True
        if _min and root.val <= _min.val:
            return False
        if _max and root.val >= _max.val:
            return False
        return self.isValidBST(root.left, _min, root) and self.isValidBST(root.right, root, _max)

    def isInBST(self, root, target):
        if not root:
            return False
        if root.val == target:
            return True
        if root.val < target:
            return self.isInBST(root.right, target)
        if root.val > target:
            return self.isInBST(root.left, target)

    def insertIntoBST(self, root, val):
        if not root:
            #return Node()
            pass
        if root.val < val:
            root.right = self.insertIntoBST(root.right, val)
        if root.val > val:
            root.left = self.insertIntoBST(root.left, val)
        return root

    def deleteNode(self, root, key):
        if not root:
            return None
        if root.val == key:
            if not root.left:
                return root.right
            if not root.right:
                return root.left
            #A 有两个⼦节点，⿇烦了，为了不破坏 BST 的性质，A 必须找到 左⼦树中最⼤的那个节点，或者右⼦树中最⼩的那个节点来接替⾃⼰
            minNode = self.getMin(root.right)
            root.val = minNode.val
            root.right = self.deleteNode(root.right, key)
        elif root.val > key:
            root.left = self.deleteNode(root.left, key)
        elif root.val < key:
            root.right = self.deleteNode(root.right, key)
        return root

    def getMin(self, node):
        while node.left:
            node = node.left
        return node

'''
解决⼀个回溯问题，实际上就是⼀个决 策树的遍历过程。你只需要思考 3 个问题： 
1、路径：也就是已经做出的选择。 
2、选择列表：也就是你当前可以做的选择。 
3、结束条件：也就是到达决策树底层，⽆法再做选择的条件

result = [] 
def backtrack(路径, 选择列表): 
    if 满⾜结束条件: result.add(路径) 
        return 
    for 选择 in 选择列表: 
        做选择 
        backtrack(路径, 选择列表) 
        撤销选择
        
其核⼼就是 for 循环⾥⾯的递归，在递归调⽤之前「做选择」，在递归调⽤ 之后「撤销选择」
'''

class MergeSort(object):
    def __init__(self):
        aux = None

    def sort(self, array=[]):
        self.aux = [0 for _ in range(len(array))]
        self.sort2(array, 0, len(array)-1)

    def sort2(self, array, lo, hi):
        pass


    def merge(self, array, lo, mid, hi):
        i = lo
        j = mid+1
        for k in range(lo, hi + 1):
            self.aux[k] = array[k]

        for k in range(lo, hi + 1):
            if i > mid:
                array[k] = self.aux[j]
                j += 1
            elif j > hi:
                array[k] = self.aux[i]
                i += 1
            elif self.aux[j] < self.aux[i]:
                array[k] = self.aux[j]
                j += 1
            else:
                array[k] = self.aux[i]
                i += 1



