exp = '''
//
int $sf = {ctr}.minFrame;
int $ef = {ctr}.maxFrame;
float $step = {ctr}.stepFrame;

int $f  = frame;

float $_rf1_ = {ctr}.cachedWeight;
float $_rf2_ = {ctr}.groundChange;
float $_rf3_ = {customCtr}.cachedToCustom;

// initial quat
float $q[4]  = {{ 0, 0, 0, 1 }};

string $cached[] = `listConnections -s off -d on ((node) + ".bkr_cachedRollingNode")`;
string $custom[] = `listConnections -s on -d off ((node) + ".bkr_customRollingNode")`;
string $bkrCtr[] = `listConnections -s on -d off ((node) + ".bkr_bkrCtrNode")`;
string $customCtr[] = `listConnections -s on -d off ((node) + ".bkr_customCtrNode")`;

$cached = sort($cached);


for($i=$sf; $i<=($f/$step); $i++)
{{
	float $t = $i * $step;
	float $qb[4] = $q;
	float $qa[3] = {{ 0, 0, 0 }};
	float $w = 0;

	float $cw = `getAttr -t $t ($bkrCtr[0] + ".cachedWeight")`;
	float $gc = `getAttr -t $t ($bkrCtr[0] + ".groundChange")`;
	float $cc = `getAttr -t $t ($customCtr[0] + ".cachedToCustom")`;
	float $ccInv = 1 - $cc;

	// ground changing weight
	$gc -= 1;
	int $cacheCnt = 1;
	int $cachedIdcs[] = {{ (int)$gc }};
	float $gcws[] = {{ 1 - ($gc - $cachedIdcs[0]) }};
	if($gcws[0] < 1)
	{{
		$gcws[1] = 1 - $gcws[0];
		$cachedIdcs[1] = $cachedIdcs[0] + 1;
		$cacheCnt++;
	}}


	for($idx=0; $idx<$cacheCnt; $idx++)
	{{
		int $k = $cachedIdcs[$idx];
		float $_q[3];
		$_q[0] = `getAttr -t $t ($cached[$k] + ".outputQuatX")`;
		$_q[1] = `getAttr -t $t ($cached[$k] + ".outputQuatY")`;
		$_q[2] = `getAttr -t $t ($cached[$k] + ".outputQuatZ")`;
		float $_w = `getAttr -t $t ($cached[$k] + ".outputQuatW")`;

		normalize($_q);
		$_w = acos($_w) * $gcws[$idx] * $ccInv;;

		for($j=0; $j<3; $j++)
			$qa[$j] += $_q[$j] * $gcws[$idx] * $ccInv;;

		$w += $_w;
	}}

	// custom rolling
	if($cc > 0)
	{{
		float $_q[3];
		$_q[0] = `getAttr -t $t ($custom[0] + ".outputQuatX")`;
		$_q[1] = `getAttr -t $t ($custom[0] + ".outputQuatY")`;
		$_q[2] = `getAttr -t $t ($custom[0] + ".outputQuatZ")`;
		float $_w = `getAttr -t $t ($custom[0] + ".outputQuatW")`;

		normalize($_q);
		$_w = acos($_w) * $cc;

		for($j=0; $j<3; $j++)
			$qa[$j] += $_q[$j] * $cc;

		$w -= $_w;
	}}

	// cached weight
	$w *= $cw;
	normalize($qa);

	for($j=0; $j<3; $j++)
		$qa[$j] *= sin($w);

	$qa[3] = cos($w);


	$q[0] =  $qa[0]*$qb[3] + $qa[1]*$qb[2] - $qa[2]*$qb[1] + $qa[3]*$qb[0];
	$q[1] = -$qa[0]*$qb[2] + $qa[1]*$qb[3] + $qa[2]*$qb[0] + $qa[3]*$qb[1];
	$q[2] =  $qa[0]*$qb[1] - $qa[1]*$qb[0] + $qa[2]*$qb[3] + $qa[3]*$qb[2];
	$q[3] = -$qa[0]*$qb[0] - $qa[1]*$qb[1] - $qa[2]*$qb[2] + $qa[3]*$qb[3];
}}

{output}.input2QuatX = $q[0];
{output}.input2QuatY = $q[1];
{output}.input2QuatZ = $q[2];
{output}.input2QuatW = $q[3];
'''
