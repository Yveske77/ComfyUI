export function calculateGrid(w, h, n) {
	let columns, rows, cellsize;

	if (w > h) {
		cellsize = h;
		columns = Math.ceil(w / cellsize);
		rows = Math.ceil(n / columns);
	} else {
		cellsize = w;
		rows = Math.ceil(h / cellsize);
		columns = Math.ceil(n / rows);
	}

	const cell_size = Math.min(w/columns, h/rows);
	return {cell_size, columns, rows};
}

export function is_all_same_aspect_ratio(imgs) {
	// assume: imgs.length >= 2
	let ratio = imgs[0].naturalWidth/imgs[0].naturalHeight;

	for(let i=1; i<imgs.length; i++) {
		let this_ratio = imgs[i].naturalWidth/imgs[i].naturalHeight;
		if(ratio != this_ratio)
			return false;
	}

	return true;
}
