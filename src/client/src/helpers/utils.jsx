// Global elements, functions, and settings such as colors.
// Basically hard-coded stuff :^)

export const DEFAULT_COLOR = "#212121"; //'#bdbdbd'
export const HIGHLIGHT_COLOR = "#f28e2b"; // '#212121'
export const DRAM_COLOR = "#5ab4ac";
export const PMEM_COLOR = "#d8b365";

export const CONFIG_LEGEND = {
	title: "Configs",
	padding: 50,
	offset: 5,
	domain: ["base", "target"],
	range: [DEFAULT_COLOR, HIGHLIGHT_COLOR],
};

export const LOC_LEGEND = {
	title: "Location",
	padding: 50,
	offset: 5,
	domain: ["DRAM", "PMEM"],
	range: [DRAM_COLOR, PMEM_COLOR],
};

export function formatPower(x) {
	let e = Math.log10(x);
	if (e !== Math.floor(e)) return; // Ignore non-exact power of ten.
	return `10${(e + "").replace(/./g, (c) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[c] || "⁻")}`;
}

export const barBisect = (scale, pos) => {
	let [ticksLeft, ticksRight] = scale.range();
	let ticksStep = scale.step();
	let ticksDomain = scale.domain();

	let val;
	if (ticksLeft < pos && pos < ticksRight) {
		for (val = 0; pos > ticksLeft + ticksStep * (val + 1); val++) {}
	}
	return ticksDomain[val];
};

export const microToSec = (msec) => {
	return Math.floor(msec / 1e6);
};

export const sortByFrequency = (array) => {
	var frequency = {};
	array.forEach(function (value) {
		frequency[value] = 0;
	});
	var uniques = array.filter(function (value) {
		return ++frequency[value] == 1;
	});
	return uniques.sort(function (a, b) {
		return frequency[b] - frequency[a];
	});
};

export const zip = (a, b) =>
	Array.from(Array(Math.max(b.length, a.length)), (_, i) => [a[i], b[i]]);

export const SERVER_URL = "http://localhost:5000";

export const CATEGORICAL_COLORS = {
	0: [0.31, 0.48, 0.65],
	1: [0.94, 0.56, 0.22],
	2: [0.87, 0.35, 0.36],
	3: [0.47, 0.72, 0.7],
	4: [0.36, 0.63, 0.32],
	5: [0.93, 0.78, 0.33],
	6: [0.69, 0.48, 0.63],
	7: [0.99, 0.62, 0.66],
	8: [0.61, 0.46, 0.38],
	9: [0.73, 0.69, 0.67],
	10: [0.31, 0.48, 0.65],
	11: [0.94, 0.56, 0.22],
	12: [0.87, 0.35, 0.36],
	13: [0.47, 0.72, 0.7],
	14: [0.36, 0.63, 0.32],
	15: [0.93, 0.78, 0.33],
	16: [0.69, 0.48, 0.63],
	17: [0.99, 0.62, 0.66],
	18: [0.61, 0.46, 0.38],
	19: [0.73, 0.69, 0.67],
};

export const rgbArrayToHex = (CMYK) => {
	let result = {};
	let c = CMYK[0];
	let m = CMYK[1];
	let y = CMYK[2];
	let k = 0;

	result.r = 1 - Math.min(1, c * (1 - k) + k);
	result.g = 1 - Math.min(1, m * (1 - k) + k);
	result.b = 1 - Math.min(1, y * (1 - k) + k);

	result.r = Math.round(result.r * 255);
	result.g = Math.round(result.g * 255);
	result.b = Math.round(result.b * 255);

	function componentToHex(c) {
		var hex = c.toString(16);
		return hex.length == 1 ? "0" + hex : hex;
	}

	return (
		"#" +
		componentToHex(result.r) +
		componentToHex(result.g) +
		componentToHex(result.b)
	);
};
