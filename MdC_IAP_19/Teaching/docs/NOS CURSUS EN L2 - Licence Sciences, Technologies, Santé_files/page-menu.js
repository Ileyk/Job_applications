var obj = null;

function checkHover() {
	if (obj) {
		obj.find('ul').fadeOut('fast');	
	} //if
} //checkHover

$j(document).ready(function() {
	$j('#Nav > li').hover(function() {
		if (obj) {
			obj.find('ul').fadeOut('fast');
			obj = null;
		} //if
		
		$j(this).find('ul').fadeIn('fast');
	}, function() {
		obj = $j(this);
		setTimeout(
			"checkHover()",
			200);
	});
	
	/*On retire la class no-js qui permet de faire dérouler le menu*/
	$j("ul#Nav").removeClass("no-js");	
});