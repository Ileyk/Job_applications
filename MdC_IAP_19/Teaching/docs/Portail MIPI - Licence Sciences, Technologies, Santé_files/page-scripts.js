var allOpen;

// Gestion affichage sMenu Accesdirect
function initDirectAccess(){

	//Au chargement : cache les menus déroulants
	$j("#submenu-directaccess").css("display","none");
	$j("#submenu-whoareyou").css("display","none");


	//Au survol : affiche sous-menus
	$j(".directaccess, #submenu-directaccess, #submenu-directaccess ul").mouseover(function(){  $j("#submenu-directaccess").css("display","block");	});
	$j(".whoareyou, #submenu-whoareyou, #submenu-whoareyou ul ").mouseover(function(){ $j("#submenu-whoareyou").css("display","block");	});
	//effets
	//$j(".directaccess").mouseover(function(){ $j("#submenu-directaccess").fadeIn("0.5"); });

	//fermeture des sous menus
	$j("#submenu-directaccess, #submenu-whoareyou, .directaccess, .whoareyou"  ).mouseout( function() { $j("#submenu-directaccess").css("display","none"); $j("#submenu-whoareyou").css("display","none"); } );

	// place les menus déroulants en position:absolute
	$j(".directaccesslinks ul").css("position","absolute");

	//Fixe hauteur du bandeaux Marron  // 1104
	if  ($j.browser.msie) {
		$j(".researchandaccess").css("height","132px");
	}
}

// ACCORDION Gestion des bloc dépilables en coeur de page
var initAccordionInitilized = false;
function initAccordion(){
	if (initAccordionInitilized) return;
	initAccordionInitilized = true;
	
	//$j(".accordion h5:first").addClass("active");
	//$j(".accordion div:not(:first)").hide();

	allOpen = false;

	leAllTxt = document.createTextNode(open_label);
	leAll = $j(document.createElement('a'));
	leAll.append(leAllTxt);
	leAll.attr('href', '#').attr('id', 'all').addClass('all_closed');
	//leAll.data('open', false);
	leAll.click(function() {
		if (allOpen)
		//if (leAll.data('open'))
		{
			$j('.accordion div').slideUp('fast');
			$j('.accordion > h5').attr("class", "inactive");
			$j(this).text(open_label);
		}
		else
		{
			$j('.accordion div').slideDown('slow');
			$j('.accordion > h5').attr("class", "active");
			$j(this).text(close_all_label);
		}
//		leAll.data('open', !leAll.data('open'));
		allOpen = !allOpen;
		return false;
	});
	
	$j(".accordion").children(':first').before(leAll);

	$j(".accordion div").hide();
	$j(".accordion > h5").toggleClass("inactive");
	
	$j(".accordion > h5").click(function(){
		/* comportement différent pour IE car bug d'affichage (Ajout d'un fadeIn) */
		if (jQuery.browser.msie) {
			if ($j(this).next("div:first")[0].style.display == "block")
			{
				$j(this).next("div:first").slideUp("fast");
			}
			else
			{
				$j(this).next("div:first").fadeIn("normal").slideDown("slow");
			}
			
			$j(this).next("div:first").siblings("div:visible").slideUp("fast");
		}
		else
		{
			$j(this).next("div:first").slideToggle("slow").siblings("div:visible").slideUp("slow");
		}
		
		if ($j(this).hasClass("active"))
		{
			$j(".accordion").children('a:first').text(open_all_label);
			allOpen = false;
		}
		$j(this).toggleClass("active");
		$j(this).toggleClass("inactive");
		
		$j(this).siblings("h5").attr("class", "inactive");
	});
}
function initSearchDirectory(){
	$j(".searchDirectory").closest("form").submit(function( event ) {
//		alert( "Handler for .submit() called." );
		var val = $j(this).find('input:radio[name=selectedSearch]:checked').val();
		if(val=="anuaire")
		{
			var searchValue = $j(this).find('input:text[name=textfield]').val();
			event.preventDefault();
			$j('<form>', {
			    "html": 
			    		'<input type="text" name="filter" value="null" />' +
			    		'<input type="text" name="form-name" value="search-person" />' +
			    		'<input type="text" name="inputPage" value="inputBadSearchWithoutFilters" />' +
			    		'<input type="text" name="name_query_type" value="NAME*" />' +
			    		'<input type="text" name="number" value="36" />' +
			    		'<input type="text" name="surname" value="" />' +
			    		'<input type="text" name="surname_query_type" value="SURNAME*" />' +
			    		'<input type="text" name="bouton.x" value="0" />' +
			    		'<input type="text" name="bouton.y" value="0" />',
			    "action": 'https://www.annuaire.upmc.fr/upmc/simpleSearch.upmc',
			    "method" : 'post'
			}).append($j('<input>').attr({
			    type: 'text',
			    name: 'name',
			    value: searchValue
			})).appendTo(document.body).submit();
		}
		else
		{
			
		}
	});
}
// ON LOAD
$j(document).ready(function(){
		initDirectAccess();
		initAccordion();
		initSearchDirectory();
		// treeview
		/*$j(".tree").treeview({
			persist: "location",
			collapsed: true,
			unique: true
		});*/
});


