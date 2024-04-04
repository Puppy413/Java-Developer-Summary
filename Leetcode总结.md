## 两数字之和

给定一个整数数组 `nums` 和一个整数目标值 `target`，请你在该数组中找出 **和为目标值** *`target`* 的那 **两个** 整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。

~~~java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        // 建立一个hashmap，key为数组值，value为数组索引
        HashMap<Integer,Integer> hashMap = new HashMap<>();
        int[] res = new int[2];
        for (int i = 0; i < nums.length; i++) {
            //遍历hashmap，如果hashmap存在另外一个加数，则往结果数组添加结果
            if (hashMap.containsKey(target-nums[i])){
                res[0] = i;
                res[1] = hashMap.get(target-nums[i]);
            }
            // 添加当前值和索引至hashmap
            hashMap.put(nums[i],i);
        }
        return  res;
    }
}
~~~

## 字母异位词分组

给你一个字符串数组，请你将 **字母异位词** 组合在一起。可以按任意顺序返回结果列表。

**字母异位词** 是由重新排列源单词的所有字母得到的一个新单词。

~~~java
 class Solution {
   public List<List<String>> groupAnagrams(String[] strs) {
    // 创建一个哈希映射，键是排序后的字符串，值是所有对应的异位词
    Map<String, List<String>> map = new HashMap<String, List<String>>();
    
    // 遍历输入的字符串数组
    for (String str : strs) {
        // 将当前字符串转换为字符数组
        char[] array = str.toCharArray();
        // 对字符数组进行排序
        Arrays.sort(array);
        // 将排序后的字符数组转换回字符串，作为哈希映射的键
        String key = new String(array);
        // 从哈希映射中获取键对应的异位词列表，如果不存在则返回一个新的列表
        List<String> list = map.getOrDefault(key, new ArrayList<String>());
        // 将当前字符串添加到异位词列表中
        list.add(str);
        // 将异位词列表放回哈希映射中
        map.put(key, list);
    }
    // 将哈希映射中的所有值（即所有的异位词列表）转换为一个列表返回
    return new ArrayList<List<String>>(map.values());
}
 }
~~~

## 最长连续序列

给定一个未排序的整数数组 `nums` ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。

请你设计并实现时间复杂度为 `O(n)` 的算法解决此问题。

~~~java
class Solution {
    public int longestConsecutive(int[] nums) {
        // 建立hashset，并将数组的元素添加至hashset，可以筛除重复元素
        Set<Integer> num_set = new HashSet<Integer>();
        for (int num : nums) {
            num_set.add(num);
        }
		// 记录结果
        int longestStreak = 0;

        for (int num : num_set) {
            // 搜索set，判断当前元素是否存在小1的元素，如果有则跳过当前循环
            if (!num_set.contains(num - 1)) {
                int currentNum = num;
                int currentStreak = 1;
				//逐渐+1，搜索连续数组
                while (num_set.contains(currentNum + 1)) {
                    currentNum += 1;
                    currentStreak += 1;
                }
				//记录当前结果最大值
                longestStreak = Math.max(longestStreak, currentStreak);
            }
        }

        return longestStreak;

    }
}
~~~

## 移动零

给定一个数组 `nums`，编写一个函数将所有 `0` 移动到数组的末尾，同时保持非零元素的相对顺序。

**请注意** ，必须在不复制数组的情况下原地对数组进行操作。

~~~java
class Solution {
    public void moveZeroes(int[] nums) {
        // 用一个索引记录最后一个非零元素
        int index = 0;
        // 覆盖数组
        for(int i = 0;i< nums.length;i++){
            if(nums[i] != 0){
                nums[index] = nums[i];
                index++;
            }
        }
        //往后填充0
        for(int i = index;i< nums.length;i++){
            nums[i] = 0;
        }
    
}
}
~~~

## 盛最多水的容器

给定一个长度为 `n` 的整数数组 `height` 。有 `n` 条垂线，第 `i` 条线的两个端点是 `(i, 0)` 和 `(i, height[i])` 。

找出其中的两条线，使得它们与 `x` 轴共同构成的容器可以容纳最多的水。

返回容器可以储存的最大水量。

**说明：**你不能倾斜容器。

~~~java
class Solution {
    public int maxArea(int[] height) {
        int max=0;
        int right = height.length-1;
        int left = 0;
        while(left<right){
            // Math.max取结果最大值，Math.min短板效应
            max = Math.max(max,Math.min(height[left],height[right])*(right-left));
            // 短板在左边则left++，短板在右边，right--
            if(height[left]<=height[right]){
                left++;
            }else{
                right--;
            }


        }
        return max;

    }
}
~~~

## 三数之和

给你一个整数数组 `nums` ，判断是否存在三元组 `[nums[i], nums[j], nums[k]]` 满足 `i != j`、`i != k` 且 `j != k` ，同时还满足 `nums[i] + nums[j] + nums[k] == 0` 。请

你返回所有和为 `0` 且不重复的三元组。

**注意：**答案中不可以包含重复的三元组。

~~~java
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
      	// 给数组排序
        Arrays.sort(nums);
        int left = 0, right = 0;
        for (int i = 0; i < nums.length; i++) {
            // 剪枝，当前元素大于0，则结果一定大于0，遍历无意义
            if (nums[i]>0){
                return result;
            }
            // 剪枝，遇到重复元素，结果相同
            if (i > 0 && nums[i] == nums[i-1] ){
                continue;
            }
            // 定义双指针
            left = i+1;
            right = nums.length-1;
            while (left<right){
                //计算三数和
                int temp = nums[i]+nums[left]+nums[right];
                //三数和偏大，right--
                if (temp>0){
                    right--;
                //三数和偏小，left++
                }else if (temp<0){
                    left++;
                //三数和为0
                }else {
                    //添加结果
                    List<Integer> list = new ArrayList<>();
                    list.add(nums[i]);
                    list.add(nums[left]);
                    list.add(nums[right]);
                    result.add(list);
                    // 剪枝，left重复，结果相同
                    while (left<right && nums[left]==nums[left+1]){
                        left++;
                    }
                    // 剪枝，right重复，结果相同
                    while (left<right && nums[right]== nums[right-1]){
                        right--;
                    }
                    left++;
                    right--;
                }
            }

        }
        return result;
    }
}
~~~

## 接雨水

给定 `n` 个非负整数表示每个宽度为 `1` 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水

~~~java
class Solution {
    public int trap(int[] height) {
        int ans = 0;
        int left = 0;
        int right = height.length-1;
        int leftMax = 0;
        int rightMax = 0;
        while(left<right){
            leftMax = Math.max(leftMax,height[left]);
            rightMax = Math.max(rightMax,height[right]);
            /**
            如果height[left]<height[right]，则必有leftMax<rightMax,则索引left接到的雨水=leftMax-height[left]
            如果height[left]>height[right]，则必有leftMax>rightMax,则索引left接到的雨水=rightMax-height[right]
            */
            if(height[left]<height[right]){
                ans += leftMax-height[left];
                left++;
            }else{
                ans += rightMax-height[right];
                right--;
            }
        }
        return ans;
    }
}
~~~

## 无重复字符最长子串

给定一个字符串 `s` ，请你找出其中不含有重复字符的最长子串的长度

~~~java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        int res = 0;
        int[] hash = new int[128];
        // i是子串的开头
        for (int i = 0, j = 0; j < s.length(); j++) {
            hash[s.charAt(j)]++;
            while (hash[s.charAt(j)] > 1) {
                hash[s.charAt(i)] --;
                i ++;
                
            }
            res = Math.max(res, j - i + 1);
        }
        return res;

    }
}
~~~

## 找到字符串所有字母的异位词

给定两个字符串 `s` 和 `p`，找到 `s` 中所有 `p` 的 **异位词** 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。

**异位词** 指由相同字母重排列形成的字符串（包括相同的字符串）。

~~~java
class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i <= s.length()-p.length(); i++) {
            if(Valid(s.substring(i,i+p.length()),p)){
                res.add(i);
            }
        }
        return res;
    }

    public boolean Valid(String sub, String p) {
        if (sub.length()!=p.length()) return false;
        int[] chars = new int[27];
        for (int i = 0; i < p.length(); i++) {
            chars[p.charAt(i) - 'a']++;
        }
        for (int i = 0; i < sub.length(); i++) {
            chars[sub.charAt(i) - 'a']--;
            if (chars[sub.charAt(i) - 'a'] < 0) return false;
        }
        return true;
    }
}
~~~

## 和为 K 的子数组

给你一个整数数组 `nums` 和一个整数 `k` ，请你统计并返回 *该数组中和为 `k` 的子数组的个数* 。

子数组是数组中元素的连续非空序列

~~~java
class Solution {
    public int subarraySum(int[] nums, int k) {
        Map<Integer,Integer> map = new HashMap<>();
        //Map<前缀和，个数>
        int n =nums.length;
        int sum = 0;
        int ret = 0;
        map.put(0,1);
        for(int i =0 ;i< n;i++){
            sum += nums[i]; 
            //如果sum-k的值等于map中的某个键，则存在和为k的子数组
            ret += map.getOrDefault(sum - k,0);

            map.put(sum,map.getOrDefault(sum,0)+1);//更新
        }
        
        
        return ret;
    }
}
~~~

## 滑动窗口最大值

给你一个整数数组 `nums`，有一个大小为 `k` 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 `k` 个数字。滑动窗口每次只向右移动一位。

返回 *滑动窗口中的最大值* 。

~~~java
public class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        if (nums == null || nums.length == 0 || k == 0) return new int[0];
        if (k == 1) return nums;

        Deque<Integer> deque = new ArrayDeque<>();
        int[] result = new int[nums.length - k + 1];
        int resultIndex = 0;

        for (int i = 0; i < nums.length; i++) {
            // 移除不在滑动窗口内的元素的索引
            while (!deque.isEmpty() && deque.peek() < i - k + 1) {
                deque.poll();
            }

            // 从队列尾部开始，移除所有小于当前元素的元素的索引
            while (!deque.isEmpty() && nums[deque.peekLast()] < nums[i]) {
                deque.pollLast();
            }

            // 将当前元素的索引添加到队列中
            deque.offer(i);

            // 当窗口的大小达到 k 时，将队列的第一个元素（即当前窗口的最大值）添加到结果数组中
            if (i >= k - 1) {
                result[resultIndex++] = nums[deque.peek()];
            }
        }

        return result;
    }
 }
~~~

## 最大子数组和

给你一个整数数组 `nums` ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

**子数组**是数组中的一个连续部分。

~~~java
class Solution {
    public int maxSubArray(int[] nums) {
        int pre = 0;
        int res = nums[0];
        for (int num : nums) {
            //如果取pre+num，则为延续，反之取num，则为断点
            pre = Math.max(pre + num, num);
            res = Math.max(res, pre);
        }
        return res;
    }

}
~~~

## 合并区间

以数组 `intervals` 表示若干个区间的集合，其中单个区间为 `intervals[i] = [starti, endi]` 。请你合并所有重叠的区间，并返回 *一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间* 。

~~~java
class Solution {
    public int[][] merge(int[][] intervals) {
        if (intervals.length == 0) {
            return new int[0][2];
        }
        // 按照第一个元素进行排序
        Arrays.sort(intervals, new Comparator<int[]>() {
            public int compare(int[] interval1, int[] interval2) {
                return interval1[0] - interval2[0];
            }
        });
        // 统计合并结果
        List<int[]> merged = new ArrayList<int[]>();
        for (int i = 0; i < intervals.length; ++i) {
            int L = intervals[i][0], R = intervals[i][1];
            //1. 若首次添加，则添加当前数组 2.若前者数组的R小于当前数组L，则中间存在空隙，添加当前数组
            if (merged.size() == 0 || merged.get(merged.size() - 1)[1] < L) {
                merged.add(new int[]{L, R});
            } else {
                // 若前者数组的R大于等于当前数组L，则更改前者数组R
                merged.get(merged.size() - 1)[1] = Math.max(merged.get(merged.size() - 1)[1], R);
            }
        }
        return merged.toArray(new int[merged.size()][]);
    }
}

~~~

## 轮转数组

给定一个整数数组 `nums`，将数组中的元素向右轮转 `k` 个位置，其中 `k` 是非负数。

~~~java
class Solution {
    public void rotate(int[] nums, int k) {
    int n = nums.length;
        int[] res = new int[n];
        for(int i = 0;i<nums.length;++i){
            //取模赋值
            res[(i+k)% nums.length] = nums[i];
        }
        //将res数组复制到nums数组中
        System.arraycopy(res, 0, nums, 0, n);

    }
}
~~~

## 除自身以外数组的乘积

给你一个整数数组 `nums`，返回 *数组 `answer` ，其中 `answer[i]` 等于 `nums` 中除 `nums[i]` 之外其余各元素的乘积* 。

题目数据 **保证** 数组 `nums`之中任意元素的全部前缀元素和后缀的乘积都在 **32 位** 整数范围内。

请 **不要使用除法，**且在 `O(*n*)` 时间复杂度内完成此题

~~~java
class Solution {
    public int[] productExceptSelf(int[] nums) {
        List<Integer> listzoro = new ArrayList<>();
        int multiply = 1;
        int n = nums.length;
        int[] anwser = new int[n];
        for (int i = 0;i < n;i++){
            //获取0值的索引
            if (nums[i] == 0){
                listzoro.add(i);
            }else{
                //统计总和
                multiply *= nums[i];
            }
        }
        if (listzoro.size() == 0){
            //数组中不存在0
            for (int i = 0;i < n;i++){
                anwser[i] = multiply / nums[i];
            }
        } else if (listzoro.size() == 1){
            //如果数组中只有一个0，0索引处则为其他值乘积
            anwser[listzoro.get(0)] = multiply;
        }
        //如果数组中有多个0，则结果全0
        return anwser;
    }

}
~~~

## 缺失的第一个正数

给你一个未排序的整数数组 `nums` ，请你找出其中没有出现的最小的正整数。

请你实现时间复杂度为 `O(n)` 并且只使用常数级别额外空间的解决方案。

~~~java
public class Solution {
    public int firstMissingPositive(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            // 通过while循环不断交换，直到无法交换为止
            while (nums[i] > 0 && nums[i] <= n && nums[nums[i] - 1] != nums[i]) {
                // 将nums[i]放到它应该在的位置，即nums[i]-1
                int temp = nums[i];
                nums[i] = nums[temp - 1];
                nums[temp - 1] = temp;
            }
        }
        
        // 再次遍历数组，查找第一个位置不正确的正数
        for (int i = 0; i < n; ++i) {
            if (nums[i] != i + 1) {
                return i + 1; // 返回应该在的位置
            }
        }
        
        // 如果数组完全正确，则返回n+1
        return n + 1;
    }
}

~~~

## 矩阵置零

给定一个 `m x n` 的矩阵，如果一个元素为 **0** ，则将其所在行和列的所有元素都设为 **0** 。请使用 **原地**算法**。**

~~~java
class Solution {
    public void setZeroes(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        HashSet<Integer> row = new HashSet<>();
        HashSet<Integer> column = new HashSet<>();
		//记录0的横纵坐标
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j]==0){
                    row.add(i);
                    column.add(j);
                }
            }
        }
        //行置零
        for (int r: row
             ) {
            for (int i = 0; i < n; i++) {
                matrix[r][i] = 0;
            }
        }
		//列置零
        for (int c: column
        ) {
            for (int i = 0; i < m; i++) {
                matrix[i][c] = 0;
            }
        }
    }
}
~~~

## 螺旋矩阵

给你一个 `m` 行 `n` 列的矩阵 `matrix` ，请按照 **顺时针螺旋顺序** ，返回矩阵中的所有元素。

~~~java
class Solution {
    public List<Integer> spiralOrder(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        int left = 0;
        int right = n-1;
        int top =0;
        int bottom =m-1;
        List<Integer> order = new ArrayList<Integer>();

        while(left<=right&&top<=bottom){
            //圈顶行遍历
            for (int column = left; column <=right ; column++) {
                order.add(matrix[top][column]);
            }
            //圈最右列遍历
            for (int row = top+1; row <= bottom ; row++) {
                order.add(matrix[row][right]);
            }
			//判断左右和上下是否重合
            if (left < right && top < bottom) {
                //圈底行遍历
                for (int column = right - 1; column > left; column--) {
                    order.add(matrix[bottom][column]);
                }
                //圈最左列遍历
                for (int row = bottom; row > top; row--) {
                    order.add(matrix[row][left]);
                }
            }
            //缩圈
            left++;
            right--;
            top++;
            bottom--;
        }
        return order;
    }
}
~~~

## 旋转图像

给定一个 *n* × *n* 的二维矩阵 `matrix` 表示一个图像。请你将图像顺时针旋转 90 度。

你必须在**原地**旋转图像，这意味着你需要直接修改输入的二维矩阵。**请不要** 使用另一个矩阵来旋转图像。

~~~java
class Solution {
    public void rotate(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
		//沿水平中线上下翻转
        for (int i = 0; i < m/2; i++) {
            int [] temp;
            temp = matrix[i];
            matrix[i] = matrix[m-1-i];
            matrix[m-1-i] = temp;
        }
		//右上 - 左下的对角线翻转
        for (int i = 0; i < m ; i++) {
            for (int j = 0; j < i; j++) {
                int temp = matrix[i][j];
                matrix[i][j]=matrix[j][i];
                matrix[j][i] = temp;
            }
        }
    }
}
~~~

## 搜索二维矩阵

编写一个高效的算法来搜索 `m x n` 矩阵 `matrix` 中的一个目标值 `target` 。该矩阵具有以下特性：

- 每行的元素从左到右升序排列。
- 每列的元素从上到下升序排列。

~~~java
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length, n = matrix[0].length;
        int row = 0, column = n-1;
        while (row < m && column >= 0){
            //从上往下遍历
            if (matrix[row][column] < target) row++;
            //从右往左遍历
            else if (matrix[row][column] > target) column--;
            else return true;
        }
        return false;
    }
}
~~~

## 相交链表

给你两个单链表的头节点 `headA` 和 `headB` ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 `null` 。

~~~java
/**
指针 pA 指向 A 链表，指针 pB 指向 B 链表，依次往后遍历
如果 pA 到了末尾，则 pA = headB 继续遍历
如果 pB 到了末尾，则 pB = headA 继续遍历
比较长的链表指针指向较短链表head时，长度差就消除了
如此，只需要将最短链表遍历两次即可找到位置
*/
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) {
            return null;
        }
        ListNode pA = headA, pB = headB;
        while (pA != pB) {
            pA = pA == null ? headB : pA.next;
            pB = pB == null ? headA : pB.next;
        }
        return pA;
    }
}

~~~

## 反转链表

给你单链表的头节点 `head` ，请你反转链表，并返回反转后的链表。

~~~java
class Solution {
    public ListNode reverseList(ListNode head) {
        ListNode pre = null;
        ListNode cur = head;
        while(cur!=null){
            //创建临时节点，指向当前节点下一个节点
            ListNode next = cur.next;
            // 当前节点的下一个节点指向上一个节点
            cur.next= pre;
            // 上一个节点指向当前节点
            pre = cur;
            //当前节点指向临时节点
            cur = next;
        }
        //最后pre指向头节点
        return pre;
    }
}
~~~

## 回文链表

给你一个单链表的头节点 `head` ，请你判断该链表是否为回文链表。如果是，返回 `true` ；否则，返回 `false` 。

~~~java
class Solution {
    public boolean isPalindrome(ListNode head) {
        List<Integer> vals = new ArrayList<Integer>();

        // 将链表的值复制到数组中
        ListNode currentNode = head;
        while (currentNode != null) {
            vals.add(currentNode.val);
            currentNode = currentNode.next;
        }

        // 使用双指针判断是否回文
        int front = 0;
        int back = vals.size() - 1;
        while (front < back) {
            if (!vals.get(front).equals(vals.get(back))) {
                return false;
            }
            front++;
            back--;
        }
        return true;
    }
}
~~~

## 环形链表

如果链表中有某个节点，可以通过连续跟踪 `next` 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 `pos` 来表示链表尾连接到链表中的位置（索引从 0 开始）。**注意：`pos` 不作为参数进行传递** 。仅仅是为了标识链表的实际情况。

*如果链表中存在环* ，则返回 `true` 。 否则，返回 `false` 。

~~~java
public class Solution {
    public boolean hasCycle(ListNode head) {
        //利用HashSet的元素唯一性，判断是否是环形列表
        Set<ListNode> seen = new HashSet<ListNode>();
        while (head != null) {
            if (!seen.add(head)) {
                return true;
            }
            head = head.next;
        }
        return false;
    }
}

public class Solution {
    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) {
            return false;
        }
        ListNode fast = head;
        ListNode slow = head;
        //快慢指针速度不一样，在环形中必定相遇，否则最后则fast.next为null
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) {
                return true;
            }
        }
        return false;
    }
}
~~~

## 环形链表II



~~~java
public class Solution {
    public ListNode detectCycle(ListNode head) {
        ListNode fast = head, slow = head;
        //判断是否是环形链表
        while (true) {
            if (fast == null || fast.next == null) return null;
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) break;
        }
        //将快指针指向链表头
        fast = head;
        while (slow != fast) {
            slow = slow.next;
            fast = fast.next;
        }
        return fast;
    }
}

~~~

## 合并有序列表

将两个升序链表合并为一个新的 **升序** 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

~~~java
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        //虚节点
        ListNode prehead = new ListNode(-1);
		//指针节点
        ListNode prev = prehead;
        while (l1 != null && l2 != null) {
            if (l1.val <= l2.val) {
                //指针节点指向l1
                prev.next = l1;
                //l1后移
                l1 = l1.next;
            } else {
                prev.next = l2;
                l2 = l2.next;
            }
            prev = prev.next;
        }

        // 合并后 l1 和 l2 最多只有一个还未被合并完，我们直接将链表末尾指向未合并完的链表即可
        prev.next = l1 == null ? l2 : l1;

        return prehead.next;
    }
}
~~~

## 两数相加

给你两个 **非空** 的链表，表示两个非负的整数。它们每位数字都是按照 **逆序** 的方式存储的，并且每个节点只能存储 **一位** 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

~~~java
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode s1 = l1;
        ListNode s2 = l2;
        while (s1 != null && s2 != null) {
            if (s1.val + s2.val >= 10) {
                if (s1.next == null) {
                    s1.next = new ListNode(1);
                } else if (s2.next == null) {
                    s2.next = new ListNode(1);
                } else {
                    s1.next.val += 1;
                }
            }
            s1.val = (s1.val + s2.val) % 10;
            s2.val = s1.val;
            s1 = s1.next;
            s2 = s2.next;
        }
        return s1 == null ? l2 : l1;
    }
}
~~~

## 删除链表的倒数第N个结点

给你一个链表，删除链表的倒数第 `n` 个结点，并且返回链表的头结点。

~~~java
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0, head);
        ListNode first = head;
        ListNode second = dummy;
        //快指针走n步，与慢指针保持n个节点的距离
        for (int i = 0; i < n; ++i) {
            first = first.next;
        }
        //快慢指针同时移动，直到快指针到末尾节点
        while (first != null) {
            first = first.next;
            second = second.next;
        }
        //删除倒数第n个结点
        second.next = second.next.next;
        ListNode ans = dummy.next;
        return ans;
    }
}
~~~

## 两两交换链表中的节点

给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。

~~~java
class Solution {
    public ListNode swapPairs(ListNode head) {
        ListNode dummyHead = new ListNode(-1,head);
        ListNode temp = dummyHead;
		
        while (temp.next != null && temp.next.next != null) {
            //第一个节点
            ListNode node1 = temp.next;
            //第二个节点
            ListNode node2 = temp.next.next;
            //临时节点下一个节点指向第二个节点
            temp.next = node2;
            //第一个节点的下一个节点指向第二个节点的下一个节点
            node1.next = node2.next;
            //第二个节点下一个节点指向第一个节点
            node2.next = node1;
            temp = node1;
        }
        return dummyHead.next;
    }
}
~~~

## K个一组反转链表

给你链表的头节点 `head` ，每 `k` 个节点一组进行翻转，请你返回修改后的链表。

`k` 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 `k` 的整数倍，那么请将最后剩余的节点保持原有顺序。

你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。

~~~java
class Solution {
    public static ListNode reverseKGroup(ListNode head, int k) {
        ListNode dummy = new ListNode(-1, head), prev = dummy;
        while (true) {
            // 检查剩余节点是否有k个，不足则返回
            ListNode last = prev;
            for (int i = 0; i < k; i++) {
                last = last.next;
                if (last == null) {
                    return dummy.next;
                }
            }

            // 翻转k个节点
            ListNode curr = prev.next, next;
            for (int i = 0; i < k - 1; i++) {
                next = curr.next;
                curr.next = next.next;
                next.next = prev.next;
                prev.next = next;
            }
            prev = curr;
        }
    }
}
~~~

